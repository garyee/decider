from alepython import ale_plot
from methodBaseClass.ImModelBase import ImModelBase
from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import Accuracy, ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType
from interpret.glassbox import LogisticRegression



class GLM(ImModelBase):
  NAME='GLM'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.VIS,ResultTypes.FI]
  CONSTRAINTS={
    'complexity': Complexity.LOW,
    'accuracy':Accuracy.MEDIUM,
    'heterogeneity':Heterogeneity.BOTH,
    'col_count':ColumCount.NONE,
    'corr_det':Correlation.NONE,
    'multicollinearity':Multicollinearity.NONE,
    'linearity':Linearity.YES,
    'monotonicity':Monotonicity.YES,
    'interactivity':Interactivity.MANUALLY,
  }

  CONSTANT = LogisticRegression()

  
 