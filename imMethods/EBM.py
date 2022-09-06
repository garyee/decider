from alepython import ale_plot
from methodBaseClass.ImModelBase import ImModelBase

from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import Accuracy, ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType
from interpret.glassbox import ExplainableBoostingClassifier


class EBM(ImModelBase):
  NAME='EBM'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.VIS,ResultTypes.FI]
  CONSTRAINTS={
    'complexity': Complexity.HIGH,
    'accuracy':Accuracy.HIGH,
    'heterogeneity':Heterogeneity.BOTH,
    'col_count':ColumCount.NONE,
    'corr_det':Correlation.NONE,
    'multicollinearity':Multicollinearity.NONE,
    'linearity':Linearity.NO,
    'monotonicity':Monotonicity.YES,
    'interactivity':Interactivity.YES,
  }
  CONSTANT = ExplainableBoostingClassifier(n_jobs=-1)
  # CONSTANT = ExplainableBoostingRegressor(n_jobs=-1)
 