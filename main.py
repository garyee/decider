
from utils.computeDataSetProperties import getDataPropPost, getDataPropPre
from utils.enums import ExplanationScope, TaskType, XaiMode
from utils.helper import getTrainAndTestSetXandY
from recomender import getRecommendation
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
    'resultType': None,
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
getDataPropPre(dataSetProperties,initialConfig,X_train, X_test, y_train, y_test)
X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test,initialConfig["taskType"])
getDataPropPost(dataSetProperties,initialConfig,X_train, X_test, y_train, y_test)

resList=getRecommendation(initialConfig,dataSetProperties)

print(resList)