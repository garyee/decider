
from interpretableModels import applyInterpretableModel
from utils.enums import ExplanationScope, TaskType, XaiMode
from utils.helper import getTrainAndTestSetXandY, getCatlAndNumColsCount
from utils.properties import printCorrMatrix
from modelAgnostic import applyPostHoc
import pandas as pd
import numpy as np

from utils.preprocessing import preprocess

X_train = pd.read_csv("./data/titanic_train.csv", dtype={"Age": np.float64}, )
X_test = None

totalColCount,catColCount,numColCount=getCatlAndNumColsCount(X_train)

# interactionValue
# correlationValue
# Mon

initialConfig={
    'targetLabel':'Survived',
    'taskType': TaskType.CLASSIFICATION,
    'explanationScope': ExplanationScope.GLOBAL,
    'xaiMode': XaiMode.INTERPRETABLE_MODEL,
    # pass in Sklearn compatible model
    'trainedModel': None
    #'feature of interrest'
}
dataSetProperties={
    'col_count':totalColCount,
    'cat_col_count':catColCount,
    'num_col_count':numColCount
}

X_train, X_test, y_train, y_test = getTrainAndTestSetXandY(X_train,X_test,initialConfig["targetLabel"])

corrMatrix = X_train.corr()
print(corrMatrix)
dataSetProperties['corr_det'] = (np.linalg.det(corrMatrix))

# X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test,initialConfig["taskType"])

# printCorrMatrix(X_train)

if initialConfig["trainedModel"] is not None:
    initialConfig["xaiMode"]=XaiMode.POST_HOC

if initialConfig["xaiMode"]==XaiMode.POST_HOC:
  applyPostHoc(X_train, X_test, y_train, y_test,initialConfig["taskType"],initialConfig["explanationScope"],initialConfig["trainedModel"])
elif initialConfig["xaiMode"]==XaiMode.INTERPRETABLE_MODEL:
  applyInterpretableModel(X_train, y_train,initialConfig.taskTypeskType)

print(dataSetProperties)