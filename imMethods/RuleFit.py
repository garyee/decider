from alepython import ale_plot
from methodBaseClass.ImModelBase import ImModelBase
from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import Accuracy, ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType
from interpret.glassbox import ClassificationTree



class RuleFit(ImModelBase):
  NAME='RuleFit'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.BOTH
  RESULTS=[ResultTypes.TEXT]
  CONSTRAINTS={
    'complexity': Complexity.HIGH,
    'accuracy':Accuracy.LOW,
    'heterogeneity':Heterogeneity.BOTH,
    'col_count':ColumCount.BAD,
    'corr_det':Correlation.BAD,
    'multicollinearity':Multicollinearity.NONE,
    'linearity':Linearity.NO,
    'monotonicity':Monotonicity.NO,
    'interactivity':Interactivity.YES,
  }

  CONSTANT = ClassificationTree()
  # CONSTANT = RegressionTree()

  
 