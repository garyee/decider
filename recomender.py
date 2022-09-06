from imMethods.ImModels import MethodClassDecisionList, MethodClassTree
from maMethods.ale import ALE
from methodBaseClass.XaiBaseClass import XAIMethod
from utils.enums import Accuracy, Complexity, ExplanationScope, TaskType, XaiMode
from utils.filter import filterResultType, filterScope, filterTaskType, filterXaiMethod


def getRecommendation(config,properties):

    pool=[MethodClassTree(),MethodClassDecisionList(),ALE()]
    filterReasons=[]

    if('indexOfTheLocalSample' in config and config['indexOfTheLocalSample'] is not None):
        config['explanationScope']=ExplanationScope.LOCAL

    if('featureOfInterrest' in config and config['featureOfInterrest'] is not None):
        config['explanationScope']=ExplanationScope.GLOBAL

    if('userModel' in config and config['userModel'] is not None):
        config['xaiMode']=XaiMode.POST_HOC

    if('xaiMode' in config and config['xaiMode'] is not None):
        pool=filterXaiMethod(pool,config['xaiMode'],filterReasons)

    if('taskType' in config and config['taskType'] is not None):
        pool=filterTaskType(pool,config['taskType'],filterReasons)
    
    if('explanationScope' in config and config['explanationScope'] is not None):
        pool=filterScope(pool,config['explanationScope'],filterReasons)

    if('resultType' in config and config['resultType'] is not None):
        pool=filterResultType(pool,config['resultType'],filterReasons)

    rankedDist=rankPool(pool,properties)


    # if "trainedModel" in initialConfig and initialConfig["trainedModel"] is not None:
#     initialConfig["xaiMode"]=XaiMode.POST_HOC

# if "xaiMode" in initialConfig and initialConfig["xaiMode"]==XaiMode.POST_HOC:
#   applyPostHoc(X_train, X_test, y_train, y_test,initialConfig["taskType"],initialConfig["explanationScope"],initialConfig["trainedModel"])
# elif "xaiMode" in initialConfig and  initialConfig["xaiMode"]==XaiMode.INTERPRETABLE_MODEL:
#   applyInterpretableModel(X_train, y_train,initialConfig["taskType"])


    return None;