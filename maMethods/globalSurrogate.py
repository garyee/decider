from interpret.ext.blackbox import MimicExplainer
from interpret.ext.glassbox import LGBMExplainableModel

from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType

class GS(PostHocBase):
  NAME='Global Surrogate'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.FI]
  CONSTRAINTS={
    'complexity': Complexity.HIGH,
    'heterogeneity':Heterogeneity.BOTH,
    'col_count':ColumCount.NONE,
    'corr_det':Correlation.NONE,
    'multicollinearity':Multicollinearity.NONE,
    'linearity':Linearity.BOTH,
    'monotonicity':Monotonicity.NO,
    'interactivity':Interactivity.NO,
  }
  def apply(model, df_train,df_test, number_of_feature_to_display=0):
    surrogate = LGBMExplainableModel
    explainer = MimicExplainer(model,
                            df_train,
                            surrogate,
                            augment_data=False,
                            features=df_train.columns)

    global_explanation = explainer.explain_global(df_test)
    fig=global_explanation.visualize()
    fig.show()