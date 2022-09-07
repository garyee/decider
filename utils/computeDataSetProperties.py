
import math
from utils.helper import getCatlAndNumColsCount
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import GradientBoostingRegressor

from utils.interactivity2 import h_stat

def getDataPropPre(propDict,config,X_train, X_test, y_train, y_test):
    total,het=computeHeterogeneity(X_train)
    propDict['heterogeneity']=het
    propDict['col_count']=total
    propDict['corr_det']=computeCorrDet(X_train)
    propDict['linearity']=computeLinearity(X_train, y_train,config['targetLabel'])
    propDict['monotonicity']=computeMonotonicity(X_train,y_train,config['targetLabel'])

def getDataPropPost(propDict,config,X_train, X_test, y_train, y_test):
    propDict['interactivity']=computeInteractivity(X_train, X_test, y_train, y_test,config['targetLabel'])
    propDict['multicollinearity']=computeMulticollinearity(X_train,y_train,config['targetLabel'])


#1 - all categorical
def computeHeterogeneity(X_train):
    totalColCount,catColCount,numColCount=getCatlAndNumColsCount(X_train)
    return totalColCount,catColCount/totalColCount

def computeCorrDet(X_train):
    corrMat=X_train.corr()
    return np.linalg.det(corrMat).round(4)

def computeMulticollinearity(X_train,y_train,targetLabel):
    fullXy=pd.concat([X_train,y_train],axis=1)
    features = "+".join(fullXy.columns)
    y, X = dmatrices(targetLabel+' ~' + features, fullXy, return_type='dataframe')
    
    vifArr = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = pd.DataFrame(vifArr)
    vif.replace([np.inf, -np.inf], np.nan, inplace=True)
    vif.dropna(inplace=True)
    return vif.mean()[0]

def computeLinearity(X_train,y_train,targetName):
    fullXy=pd.concat([X_train,y_train],axis=1)
    corrMat=fullXy.corr()
    targetCol=corrMat.loc[:, [targetName]]
    targetCol.drop(targetName,inplace=True)
    return abs(targetCol.mean()[targetName])

def computeMonotonicity(X_train,y_train,targetName):
    fullXy=pd.concat([X_train,y_train],axis=1)
    corrMat=fullXy.corr(method="spearman")
    targetCol=corrMat.loc[:, [targetName]]
    targetCol.drop(targetName,inplace=True)
    return targetCol.mean()[targetName]

#ANOVA
# def computeInteractivity(X_train, X_test, y_train, y_test,targetName):
#     fullXy=pd.concat([X_train,y_train],axis=1)
#     features = "+".join(fullXy.columns)
#     model = ols(targetName+' ~' + features, data=fullXy).fit()
#     tmp=sm.stats.anova_lm(model, typ=2)
#     print(tmp)

#     interActionMatrix = pd.DataFrame(tmp)
#     # print(interActionMatrix)
#     return 0

def computeInteractivity(X_train, X_test, y_train, y_test,targetName):
    gbr_1 = GradientBoostingRegressor(max_depth=10,random_state = 42)
    gbr_1.fit(X_train, y_train)
    print('GBM: '+str(gbr_1.score(X_test, y_test)))
    hstat=h_stat(gbr_1, X_train, 'all')
    print('H: '+hstat)
#     # if(math.isnan(hstat)):
#     #     return 0
#     # return hstat
    return 0

