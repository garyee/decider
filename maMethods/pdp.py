from sklearn.inspection import PartialDependenceDisplay

from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import ExplanationScope, ResultTypes, TaskType

class PDP(PostHocBase):
  NAME='PDP'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.VIS]
  CONSTRAINTS={
    'heterogeneity':None,
    'col_count':None,
    'corr_det':None,
    'multicollinearity':None,
    'linearity':None,
    'monotonicity':None,
    'interactivity':None,
  }

  def apply(model,df_train,number_of_feature_to_display=0):
    pdp = PartialDependenceDisplay.from_estimator(model,       
                                    df_train,
                                    features=[number_of_feature_to_display],
                                    feature_names=df_train.columns.values.tolist(),
                                    grid_resolution=100,
                                    ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
                                    pd_line_kw={"color": "tab:orange", "linestyle": "--"},
                                    kind='average')