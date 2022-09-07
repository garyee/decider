from alepython import ale_plot
from methodBaseClass.ImModelBase import ImModelBase
from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import Accuracy, ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType
from interpret.glassbox import LogisticRegression



class LogReg(ImModelBase):
  NAME='Logistic regression'
  TASKTYPE=TaskType.CLASSIFICATION
  SCOPE=ExplanationScope.GLOBAL
  RESULTS=[ResultTypes.VIS,ResultTypes.FI]
  CONSTRAINTS={
    'complexity': Complexity.LOW,
    'accuracy':Accuracy.LOW,
    'heterogeneity':Heterogeneity.NUMMERICAL,
    'col_count':ColumCount.BAD,
    'corr_det':Correlation.BAD,
    'multicollinearity':Multicollinearity.VERYBAD,
    'linearity':Linearity.YES,
    'monotonicity':Monotonicity.YES,
    'interactivity':Interactivity.MANUALLY,
  }

  CONSTANT = LogisticRegression()

  
 