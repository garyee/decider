from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from methodBaseClass.PostHocBase import PostHocBase

from utils.enums import ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType

class PFI(PostHocBase):
  NAME='PFI'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.FI]
  CONSTRAINTS={
    'complexity': Complexity.LOW,
    'heterogeneity':Heterogeneity.BOTH,
    'col_count':ColumCount.NONE,
    'corr_det':Correlation.BAD,
    'multicollinearity':Multicollinearity.NONE,
    'linearity':Linearity.YES,
    'monotonicity':Monotonicity.NO,
    'interactivity':Interactivity.NO,
  }

  def apply(model,df_train,df_train_y):
      result = permutation_importance(model,
                                      df_train,
                                      df_train_y,
                                      n_repeats=10,
                                      random_state=42,
                                      n_jobs=-1)
      sorted_idx = result.importances_mean.argsort()
      fig, ax = plt.subplots()
      fpi=ax.boxplot(result.importances[sorted_idx].T,vert=False, labels=df_train.columns[sorted_idx])
      ax.set_title("Permutation Importances Titanic dataset (training set)")
      fig.tight_layout()
      plt.show()