from alepython import ale_plot
from methodBaseClass.ImModelBase import ImModelBase
from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import Accuracy, ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType
from interpret.glassbox import ClassificationTree



class Tree(ImModelBase):
  NAME='Classification Tree'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.VIS,ResultTypes.FI]
  CONSTRAINTS={
    'complexity': Complexity.MEDIUM,
    'accuracy':Accuracy.MEDIUM,
    'heterogeneity':Heterogeneity.CATEGORICAL,
    'col_count':ColumCount.NONE,
    'corr_det':Correlation.NONE,
    'multicollinearity':Multicollinearity.NONE,
    'linearity':Linearity.YES,
    'monotonicity':Monotonicity.NO,
    'interactivity':Interactivity.NO,
  }

  CONSTANT = ClassificationTree()
  # CONSTANT = RegressionTree()

  
 