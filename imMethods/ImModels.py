from interpret.glassbox import ExplainableBoostingRegressor, LogisticRegression, LinearRegression, RegressionTree
from interpret.glassbox import ExplainableBoostingClassifier, ClassificationTree, DecisionListClassifier

from methodBaseClass.ImModelBase import ImModelBase


# class MethodClassDecisionList(ImModelBase):
#   CONSTRAINTS={
    
#   }
#   CONSTANT = DecisionListClassifier()

# class  MethodClassGAM(ImModelBase):
#   CONSTANT = (n_jobs=-1)


# # https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515
# class MethodRegGAM(ImModelBase):
#   CONSTANT = (n_jobs=-1)