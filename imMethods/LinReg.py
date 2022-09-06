from alepython import ale_plot
from methodBaseClass.ImModelBase import ImModelBase
from methodBaseClass.PostHocBase import PostHocBase
from utils.enums import Accuracy, ColumCount, Complexity, Correlation, ExplanationScope, Heterogeneity, Interactivity, Linearity, Monotonicity, Multicollinearity, ResultTypes, TaskType
from interpret.glassbox import LinearRegression



class LinRes(ImModelBase):
  NAME='Linear regression'
  TASKTYPE=TaskType.REGRESSION
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

  CONSTANT = LinearRegression()

  
 