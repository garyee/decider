from alepython import ale_plot
from methodBaseClass.ImModelBase import ImModelBase
from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import Accuracy, ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType
from interpret.glassbox import LogisticRegression



class GAM(ImModelBase):
  NAME='GAM'
  TASKTYPE=TaskType.BOTH
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.VIS,ResultTypes.FI]
  CONSTRAINTS={
    'complexity': Complexity.LOW,
    'accuracy':Accuracy.MEDIUM,
    'heterogeneity':Heterogeneity.BOTH,
    'col_count':ColumCount.NONE,
    'corr_det':Correlation.NONE,
    'multicollinearity':Multicollinearity.BAD,
    'linearity':Linearity.NO,
    'monotonicity':Monotonicity.YES,
    'interactivity':Interactivity.MANUALLY,
  }

  CONSTANT = LogisticRegression()

  
 