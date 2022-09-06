from interpret.ext.blackbox import MimicExplainer
from interpret.ext.glassbox import LGBMExplainableModel

from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import ExplanationScope, ResultTypes, TaskType

class GS(PostHocBase):
  NAME='Global Surrogate'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.FI]
  CONSTRAINTS={
    'heterogeneity':None,
    'col_count':None,
    'corr_det':None,
    'multicollinearity':None,
    'linearity':None,
    'monotonicity':None,
    'interactivity':None,
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