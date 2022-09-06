from sklearn.inspection import PartialDependenceDisplay

from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType

class ICE(PostHocBase):
  NAME='ICE'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.LOCAL
  RESULTS=[ResultTypes.VIS]
  CONSTRAINTS={
    'complexity': Complexity.HIGH,
    'heterogeneity':Heterogeneity.BOTH,
    'col_count':ColumCount.NONE,
    'corr_det':Correlation.VERYBAD,
    'multicollinearity':Multicollinearity.NONE,
    'linearity':Linearity.YES,
    'monotonicity':Monotonicity.YES,
    'interactivity':Interactivity.NO,
  }

  def apply(model,df_train,number_of_feature_to_display=0):
    pdp = PartialDependenceDisplay.from_estimator(model,       
                                    df_train,
                                    features=[number_of_feature_to_display],
                                    feature_names=df_train.columns.values.tolist(),
                                    grid_resolution=100,
                                    ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
                                    pd_line_kw={"color": "tab:orange", "linestyle": "--"},
                                    kind='individual')