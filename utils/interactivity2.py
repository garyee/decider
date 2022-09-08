from tqdm import tqdm
import itertools
from pdpbox.pdp_calc_utils import _calc_ice_lines_inter
from pdpbox.pdp import pdp_isolate, PDPInteract
from pdpbox.utils import (_check_model, _check_dataset, _check_percentile_range, _check_feature,
                    _check_grid_type, _check_memory_limit, _make_list,
                    _calc_memory_usage, _get_grids, _get_grid_combos, _check_classes)
from joblib import Parallel, delayed
import math
import numpy as np
import pandas as pd


def pdp_multi_interact(model, dataset, model_features, features, 
                    num_grid_points=None, grid_types=None, percentile_ranges=None, grid_ranges=None, cust_grid_points=None, 
                    cust_grid_combos=None, use_custom_grid_combos=False,
                    memory_limit=0.5, n_jobs=1, predict_kwds=None, data_transformer=None):

    def _expand_default(x, default, length):
        if x is None:
            return [default] * length
        return x

    def _get_grid_combos(feature_grids, feature_types):
        grids = [list(feature_grid) for feature_grid in feature_grids]
        for i in range(len(feature_types)):
            if feature_types[i] == 'onehot':
                grids[i] = np.eye(len(grids[i])).astype(int).tolist()
        return np.stack(np.meshgrid(*grids), -1).reshape(-1, len(grids))

    if predict_kwds is None:
        predict_kwds = dict()

    nr_feats = len(features)

    # check function inputs
    n_classes, predict = _check_model(model=model)
    _check_dataset(df=dataset)
    _dataset = dataset.copy()

    # prepare the grid
    pdp_isolate_outs = []
    if use_custom_grid_combos:
        grid_combos = cust_grid_combos
        feature_grids = []
        feature_types = []
    else:
        num_grid_points = _expand_default(x=num_grid_points, default=10, length=nr_feats)
        grid_types = _expand_default(x=grid_types, default='percentile', length=nr_feats)
        for i in range(nr_feats):
            _check_grid_type(grid_type=grid_types[i])

        percentile_ranges = _expand_default(x=percentile_ranges, default=None, length=nr_feats)
        for i in range(nr_feats):
            _check_percentile_range(percentile_range=percentile_ranges[i])

        grid_ranges = _expand_default(x=grid_ranges, default=None, length=nr_feats)
        cust_grid_points = _expand_default(x=cust_grid_points, default=None, length=nr_feats)

        _check_memory_limit(memory_limit=memory_limit)

        pdp_isolate_outs = []
        for idx in range(nr_feats):
            pdp_isolate_out = pdp_isolate(
                model=model, dataset=_dataset, model_features=model_features, feature=features[idx],
                num_grid_points=num_grid_points[idx], grid_type=grid_types[idx], percentile_range=percentile_ranges[idx],
                grid_range=grid_ranges[idx], cust_grid_points=cust_grid_points[idx], memory_limit=memory_limit,
                n_jobs=n_jobs, predict_kwds=predict_kwds, data_transformer=data_transformer)
            pdp_isolate_outs.append(pdp_isolate_out)

        if n_classes > 2:
            feature_grids = [pdp_isolate_outs[i][0].feature_grids for i in range(nr_feats)]
            feature_types = [pdp_isolate_outs[i][0].feature_type  for i in range(nr_feats)]
        else:
            feature_grids = [pdp_isolate_outs[i].feature_grids for i in range(nr_feats)]
            feature_types = [pdp_isolate_outs[i].feature_type  for i in range(nr_feats)]

        grid_combos = _get_grid_combos(feature_grids, feature_types)

    feature_list = []
    for i in range(nr_feats):
        feature_list.extend(_make_list(features[i]))

    # Parallel calculate ICE lines
    true_n_jobs = _calc_memory_usage(
        df=_dataset, total_units=len(grid_combos), n_jobs=n_jobs, memory_limit=memory_limit)

    print('jehtweidor1')
    grid_results = Parallel(n_jobs=true_n_jobs)(delayed(_calc_ice_lines_inter)(
        grid_combo, data=_dataset, model=model, model_features=model_features, n_classes=n_classes,
        feature_list=feature_list, predict_kwds=predict_kwds, data_transformer=data_transformer)
                                                for grid_combo in grid_combos)
    print('jehtweidor')
    ice_lines = pd.concat(grid_results, axis=0).reset_index(drop=True)
    pdp = ice_lines.groupby(feature_list, as_index=False).mean()

    # combine the final results
    pdp_interact_params = {'n_classes': n_classes, 
                        'features': features, 
                        'feature_types': feature_types,
                        'feature_grids': feature_grids}
    if n_classes > 2:
        pdp_interact_out = []
        for n_class in range(n_classes):
            _pdp = pdp[feature_list + ['class_%d_preds' % n_class]].rename(
                columns={'class_%d_preds' % n_class: 'preds'})
            pdp_interact_out.append(
                PDPInteract(which_class=n_class,
                            pdp_isolate_outs=[pdp_isolate_outs[i][n_class] for i in range(nr_feats)],
                            pdp=_pdp, **pdp_interact_params))
    else:
        pdp_interact_out = PDPInteract(
            which_class=None, pdp_isolate_outs=pdp_isolate_outs, pdp=pdp, **pdp_interact_params)

    return pdp_interact_out

def center(arr): return arr - np.mean(arr)

def compute_f_vals(mdl, X, features, selectedfeatures, num_grid_points=10, use_data_grid=False):
    f_vals = {}
    data_grid = None
    if use_data_grid:
        data_grid = X[selectedfeatures].values
    # Calculate partial dependencies for full feature set
    p_full = pdp_multi_interact(mdl, X, features, selectedfeatures, 
                                num_grid_points=[num_grid_points] * len(selectedfeatures),
                                cust_grid_combos=data_grid,
                                use_custom_grid_combos=use_data_grid)
    f_vals[tuple(selectedfeatures)] = center(p_full.pdp.preds.values)
    grid = p_full.pdp.drop('preds', axis=1)
    # Calculate partial dependencies for [1..SFL-1]
    for n in range(1, len(selectedfeatures)):
        for subsetfeatures in itertools.combinations(selectedfeatures, n):
            if use_data_grid:
                data_grid = X[list(subsetfeatures)].values
            print('calc pdp2')
            p_partial = pdp_multi_interact(mdl, X, features, subsetfeatures, 
                                        num_grid_points=[num_grid_points] * len(selectedfeatures),
                                        cust_grid_combos=data_grid,
                                        use_custom_grid_combos=use_data_grid)
            p_joined = pd.merge(grid, p_partial.pdp, how='left')
            f_vals[tuple(subsetfeatures)] = center(p_joined.preds.values)
    return f_vals

#orig
# def compute_h_val(f_vals, selectedfeatures):
#     denom_els = f_vals[tuple(selectedfeatures)].copy()
#     numer_els = f_vals[tuple(selectedfeatures)].copy()
#     sign = -1.0
#     for n in range(len(selectedfeatures)-1, 0, -1):
#         for subfeatures in itertools.combinations(selectedfeatures, n):
#             numer_els += sign * f_vals[tuple(subfeatures)]
#         sign *= -1.0
#     numer = np.sum(numer_els**2)
#     denom = np.sum(denom_els**2)
#     return math.sqrt(numer/denom) if numer < denom else np.nan

def compute_h_val_any(f_vals, allfeatures, selectedfeature):
    otherfeatures = list(allfeatures)
    otherfeatures.remove(selectedfeature)
    denom_els = f_vals[tuple(allfeatures)].copy()
    numer_els = denom_els.copy()
    numer_els -= f_vals[(selectedfeature,)]
    numer_els -= f_vals[tuple(otherfeatures)]
    numer = np.sum(numer_els**2)
    denom = np.sum(denom_els**2)
    return math.sqrt(numer/denom) if numer < denom else np.nan

#from1
def unique_rows_with_counts(inp_arr):
    width = inp_arr.shape[1]
    cont_arr = np.ascontiguousarray(inp_arr)
    tuple_dtype = [(str(i), inp_arr.dtype) for i in range(width)]
    tuple_arr = cont_arr.view(tuple_dtype)
    uniq_arr, counts = np.unique(tuple_arr, return_counts = True)
    outp_arr = uniq_arr.view(inp_arr.dtype).reshape(-1, width)
    return outp_arr, counts

#from1
def compute_h_val(f_vals, arr, inds):
    feat_vals, feat_val_counts = unique_rows_with_counts(arr)
    uniq_height = feat_vals.shape[0]
    numer_els = np.zeros(uniq_height)
    denom_els = np.empty_like(numer_els)
    for i in range(uniq_height):
        feat_vals_i = feat_vals[i]
        sign = 1.0
        for n in range(len(inds), 0, -1):
            for subinds in itertools.combinations(inds, n):
                numer_els[i] += sign*f_vals[subinds][tuple(feat_vals_i[(subinds,)])]
            sign *= -1.0
        denom_els[i] = f_vals[inds][tuple(feat_vals_i[(inds,)])]
    numer = np.dot(feat_val_counts, numer_els**2)
    denom = np.dot(feat_val_counts, denom_els**2)
    return math.sqrt(numer/denom) if numer < denom else np.nan



def h_stat(gbm, array_or_frame, indices_or_columns = 'all'):
    if indices_or_columns == 'all':
        if gbm.max_depth < array_or_frame.shape[1]:
            raise \
                Exception(
                    "gbm.max_depth == {} < array_or_frame.shape[1] == {}, so indices_or_columns must not be 'all'."
                    .format(gbm.max_depth, array_or_frame.shape[1])
                )
    else:
        if gbm.max_depth < len(indices_or_columns):
            raise \
                Exception(
                    "gbm.max_depth == {}, so indices_or_columns must contain at most {} {}."
                    .format(gbm.max_depth, gbm.max_depth, "element" if gbm.max_depth == 1 else "elements")
                )
    check_args_contd(array_or_frame, indices_or_columns)

    arr, model_inds = get_arr_and_model_inds(array_or_frame, indices_or_columns)

    width = arr.shape[1]
    f_vals = {}
    for n in tqdm(range(width, 0, -1)):
        for inds in itertools.combinations(range(width), n):
            f_vals[inds] = compute_f_vals(gbm, array_or_frame, array_or_frame.columns, array_or_frame.columns)

    return compute_h_val(f_vals, arr, tuple(range(width)))


def check_args_contd(array_or_frame, indices_or_columns):
    if indices_or_columns != 'all':
        if len(indices_or_columns) < 2:
            raise Exception("indices_or_columns must be 'all' or contain at least 2 elements.")
        if isinstance(array_or_frame, np.ndarray):
            all_inds = range(array_or_frame.shape[1])
            if not all(ind in all_inds for ind in indices_or_columns):
                raise Exception("indices_or_columns must be 'all' or a subset of {}.".format(all_inds))
        else:
            all_cols = array_or_frame.columns.tolist()
            if not all(col in all_cols for col in indices_or_columns):
                raise Exception("indices_or_columns must be 'all' or a subset of {}.".format(all_cols))

def get_arr_and_model_inds(array_or_frame, indices_or_columns):
    if isinstance(array_or_frame, np.ndarray):
        if indices_or_columns == 'all': indices_or_columns = range(array_or_frame.shape[1])
        arr = array_or_frame[:, indices_or_columns]
        model_inds = np.array(indices_or_columns)
    else:
        all_cols = array_or_frame.columns.tolist()
        if indices_or_columns == 'all': indices_or_columns = all_cols
        arr = array_or_frame[indices_or_columns].values
        model_inds = np.array([all_cols.index(col) for col in indices_or_columns])
    return arr, model_inds