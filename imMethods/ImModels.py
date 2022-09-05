from imMethods.ImModelABC import XAIMethod
from interpret.glassbox import ExplainableBoostingRegressor, LogisticRegression, LinearRegression, RegressionTree
from interpret.glassbox import ExplainableBoostingClassifier, ClassificationTree, DecisionListClassifier


class MethodClassTree(XAIMethod):
  CONSTRAINTS={
    
  }
  CONSTANT = ClassificationTree()

class MethodClassDecisionList(XAIMethod):
  CONSTRAINTS={
    
  }
  CONSTANT = DecisionListClassifier()

class MethodClassEBM(XAIMethod):
  CONSTRAINTS={
    
  }
  CONSTANT = ExplainableBoostingClassifier(n_jobs=-1)

# class  MethodClassGAM(XAIMethod):
#   CONSTANT = (n_jobs=-1)

class MethodRegLog(XAIMethod):
  CONSTRAINTS={
    
  }
  CONSTANT = LogisticRegression()

class MethodRegLinear(XAIMethod):
  CONSTRAINTS={
    
  }
  CONSTANT = LinearRegression()

class MethodRegTree(XAIMethod):
  CONSTRAINTS={
    
  }
  CONSTANT = RegressionTree()

class MethodRegEBM(XAIMethod):
  CONSTRAINTS={
    
  }
  CONSTANT = ExplainableBoostingRegressor(n_jobs=-1)

# # https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515
# class MethodRegGAM(XAIMethod):
#   CONSTANT = (n_jobs=-1)