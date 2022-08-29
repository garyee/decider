from sklearn.inspection import PartialDependenceDisplay


def applyICE(model,df_train,number_of_feature_to_display=0):
  pdp = PartialDependenceDisplay.from_estimator(model,       
                                  df_train,
                                  features=[number_of_feature_to_display],
                                  feature_names=df_train.columns.values.tolist(),
                                  grid_resolution=100,
                                  ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
                                  pd_line_kw={"color": "tab:orange", "linestyle": "--"},
                                  kind='both')