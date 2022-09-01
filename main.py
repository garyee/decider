
from interpretableModels import applyInterpretableModel
from utils.enums import ExplanationScope, TaskType, XaiMode
from utils.helper import getTrainAndTestSetXandY
from utils.properties import printCorrMatrix
from modelAgnostic import applyPostHoc
import pandas as pd
import numpy as np

from utils.preprocessing import preprocess

X_train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
X_test = pd.read_csv("/test.csv", dtype={"Age": np.float64}, )

initialConfig={
    'targetLabel':'survived',
    'taskType':TaskType.CLASSIFICATION,
    'explanationScope': ExplanationScope.GLOBAL,
    'xaiMode': XaiMode.INTERPRETABLE_MODEL,
}

X_train, X_test, y_train, y_test = getTrainAndTestSetXandY(X_train,X_test,initialConfig.targetLabel)
X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test,initialConfig.taskType)

printCorrMatrix(X_train)

if initialConfig.xaiMode==XaiMode.POST_HOC:
  applyPostHoc(X_train, X_test, y_train, y_test,initialConfig.taskType,initialConfig.explanationScope)
elif initialConfig.xaiMode==XaiMode.INTERPRETABLE_MODEL:
  applyInterpretableModel(X_train, y_train,initialConfig.taskTypeskType)