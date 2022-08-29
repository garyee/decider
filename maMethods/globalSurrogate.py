from interpret.ext.blackbox import MimicExplainer
from interpret.ext.glassbox import LGBMExplainableModel

def applyGlobalSurrogate(model, df_train,df_test, number_of_feature_to_display=0):
  surrogate = LGBMExplainableModel
  explainer = MimicExplainer(model,
                           df_train,
                           surrogate,
                           augment_data=False,
                           features=df_train.columns)

  global_explanation = explainer.explain_global(df_test)
  fig=global_explanation.visualize()
  fig.show()