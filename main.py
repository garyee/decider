
from interpretableModels import applyInterpretableModel
from utils.computeDataSetProperties import getDataPropPost, getDataPropPre
from utils.enums import ExplanationScope, TaskType, XaiMode
from utils.helper import getTrainAndTestSetXandY
from utils.properties import printCorrMatrix
from modelAgnostic import applyPostHoc
import pandas as pd
import numpy as np

from utils.preprocessing import preprocess

X_train = pd.read_csv("./data/titanic_train.csv", dtype={"Age": np.float64}, )
X_test = None


# interactionValue
# correlationValue
# Mon

initialConfig={
    'targetLabel':'Survived',
    'taskType': TaskType.CLASSIFICATION,
    'explanationScope': ExplanationScope.GLOBAL,
    'xaiMode': XaiMode.INTERPRETABLE_MODEL,
    # pass in Sklearn compatible model
    'userModel': None,
    #'feature of interrest'
    'featureOfInterrest':None,
    #indexOfTheLocalSample
    'indexOfTheLocalSample':None,
}
dataSetProperties={
    'heterogeneity':None,
    'col_count':None,
    'corr_det':None,
    'multicollinearity':None,
    'linearity':None,
    'monotonicity':None,
    'interactivity':None,
}

X_train, X_test, y_train, y_test = getTrainAndTestSetXandY(X_train,X_test,initialConfig["targetLabel"])
getDataPropPre(dataSetProperties,initialConfig,X_train,y_train)
X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test,initialConfig["taskType"])
getDataPropPost(dataSetProperties,initialConfig,X_train,y_train)


# if "trainedModel" in initialConfig and initialConfig["trainedModel"] is not None:
#     initialConfig["xaiMode"]=XaiMode.POST_HOC

# if "xaiMode" in initialConfig and initialConfig["xaiMode"]==XaiMode.POST_HOC:
#   applyPostHoc(X_train, X_test, y_train, y_test,initialConfig["taskType"],initialConfig["explanationScope"],initialConfig["trainedModel"])
# elif "xaiMode" in initialConfig and  initialConfig["xaiMode"]==XaiMode.INTERPRETABLE_MODEL:
#   applyInterpretableModel(X_train, y_train,initialConfig["taskType"])

print(dataSetProperties)