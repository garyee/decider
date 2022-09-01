import lime
import lime.lime_tabular
from interpret_community.tabular_explainer import TabularExplainer
import numpy as np

from utils.enums import TaskType

def applyLime(model,df_train,df_test,df_train_y,taskType,index=42):
  explainer = lime.lime_tabular.LimeTabularExplainer(
      np.array(df_train),
      mode = (taskType==TaskType.REGRESSION and 'regression') or 'classification',
      feature_names=np.array(df_train.columns),
      class_names=df_train_y.unique(),
      discretize_continuous=True
  )
  i = np.random.randint(0, df_test.shape[0])
  exp = explainer.explain_instance(
      np.array(df_train)[i],
        (taskType==TaskType.CLASSIFICATION and model.predict) or model.predict_proba,
        num_features=5
  )
  exp.show_in_notebook(show_all=True)