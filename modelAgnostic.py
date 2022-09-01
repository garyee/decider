from maMethods.blackBoxModel import trainBlackBoxModel
from utils.enums import ExplanationScope
from maMethods.globalSurrogate import applyGlobalSurrogate
from maMethods.lime import applyLime


def applyPostHoc(X_train,X_test,y_train,y_test,taskType,scope):
  model=trainBlackBoxModel(X_train, y_train,X_test,y_test,taskType)
  if(scope==ExplanationScope.LOCAL):
    localMethod(model,X_train,X_test,y_train,taskType)
  else:
    globalMethod(model,X_train,X_test,y_train,y_test)

def globalMethod(model,df_train,df_test,df_train_y,df_test_y):
    applyGlobalSurrogate(model, df_train,df_test)
    #applyPFI(model,df_train,df_train_y)
    #applyALE(model,df_train,)
    #applyICE(model,df_train,df_test)
    #applySHAP(model,df_train,df_test)

def localMethod(model,df_train,df_test,df_train_y,taskType):
    applyLime(model,df_train,df_test,df_train_y,taskType)