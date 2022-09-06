from interpret.glassbox import ExplainableBoostingRegressor, LogisticRegression, LinearRegression, RegressionTree
from interpret.glassbox import ExplainableBoostingClassifier, ClassificationTree, DecisionListClassifier

from methodBaseClass.ImModelBase import ImModelBase


class MethodClassTree(ImModelBase):
  NAME='classificationTree'
  CONSTRAINTS={
    'heterogeneity':None,
    'col_count':None,
    'corr_det':None,
    'multicollinearity':None,
    'linearity':None,
    'monotonicity':None,
    'interactivity':None,
  }
  CONSTANT = ClassificationTree()

class MethodClassDecisionList(ImModelBase):
  CONSTRAINTS={
    
  }
  CONSTANT = DecisionListClassifier()

class MethodClassEBM(ImModelBase):
  CONSTRAINTS={
    
  }
  CONSTANT = ExplainableBoostingClassifier(n_jobs=-1)

# class  MethodClassGAM(ImModelBase):
#   CONSTANT = (n_jobs=-1)

class MethodRegLog(ImModelBase):
  CONSTRAINTS={
    
  }
  CONSTANT = LogisticRegression()

class MethodRegLinear(ImModelBase):
  CONSTRAINTS={
    
  }
  CONSTANT = LinearRegression()

class MethodRegTree(ImModelBase):
  CONSTRAINTS={
    
  }
  CONSTANT = RegressionTree()

class MethodRegEBM(ImModelBase):
  CONSTRAINTS={
    
  }
  CONSTANT = ExplainableBoostingRegressor(n_jobs=-1)

# # https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515
# class MethodRegGAM(ImModelBase):
#   CONSTANT = (n_jobs=-1)