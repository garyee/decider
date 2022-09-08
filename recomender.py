from imMethods.EBM import EBM
from imMethods.GAM import GAM
from imMethods.GLM import GLM
from imMethods.LinReg import LinReg
from imMethods.LogReg import LogReg
from imMethods.RuleFit import RuleFit
from imMethods.SkopeRules import SkopeRules
from imMethods.Tree import Tree
from maMethods.ale import ALE
from maMethods.globalSurrogate import GS
from maMethods.ice import ICE
from maMethods.lime import LIME
from maMethods.pdp import PDP
from maMethods.pfi import PFI
from maMethods.shap import SHAP
from methodBaseClass.XaiBaseClass import XAIMethod
from utils.enums import Accuracy, Complexity, ExplanationScope, TaskType, XaiMode
from utils.filter import filterResultType, filterScope, filterTaskType, filterXaiMethod, rankPool


def getRecommendation(config,properties):

    pool=[EBM(),GAM(),GLM(),LinReg(),LogReg(),RuleFit(),SkopeRules(),Tree(),ALE(),GS(),ICE(),LIME(),PDP(),PFI(),SHAP()]
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
    res=[]
    index=0
    rankedDist = dict(sorted(rankedDist.items(), key=lambda x:int(x[0]),reverse=True))
    for key, value in rankedDist.items():
        if(index>4):
            break
        if type(value) == list:
            for  innerValue in value:
                if(index>4):
                    break
                res.append(innerValue.getName()+' (Score: '+str(key)+')')
                index+=1
        else:
            res.append(value.getName()+' (Score: '+str(key)+')')
            index+=1


    # if "trainedModel" in initialConfig and initialConfig["trainedModel"] is not None:
#     initialConfig["xaiMode"]=XaiMode.POST_HOC

# if "xaiMode" in initialConfig and initialConfig["xaiMode"]==XaiMode.POST_HOC:
#   applyPostHoc(X_train, X_test, y_train, y_test,initialConfig["taskType"],initialConfig["explanationScope"],initialConfig["trainedModel"])
# elif "xaiMode" in initialConfig and  initialConfig["xaiMode"]==XaiMode.INTERPRETABLE_MODEL:
#   applyInterpretableModel(X_train, y_train,initialConfig["taskType"])

    return res;