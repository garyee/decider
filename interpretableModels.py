from utils.enums import TaskType
from imMethods.ImModels import MethodClassEBM,MethodClassTree,MethodClassDecisionList,MethodRegLog,MethodRegLinear,MethodRegTree,MethodRegEBM

def applyInterpretableModel(X_train, y_train,taskType):
  if(taskType==TaskType.CLASSIFICATION):
    trainGlassBoxClassMLModel(X_train, y_train)
  else:
    trainGlassBoxRegMLModel(X_train, y_train)

def trainGlassBoxRegMLModel(df_train, df_y_train,model=None):
  if(model=='logistic'):
    MethodRegLog(df_train, df_y_train)
  elif(model=='linear'):
    MethodRegLinear(df_train, df_y_train)
  elif(model=='decisionTree'):
    MethodRegTree(df_train, df_y_train)
  # elif(model=='GAM'):
  #   MethodRegGAM(df_train, df_y_train)
  else:
    MethodRegEBM(df_train, df_y_train)

def trainGlassBoxClassMLModel(df_train, df_y_train,model=None):
  if(model=='decisionTree'):
    MethodClassTree(df_train, df_y_train)
  elif(model=='rules'):
    MethodClassDecisionList(df_train, df_y_train)
  # elif(model=='GAM'):
  #   MethodClassGAM(df_train, df_y_train)
  else:
    MethodClassEBM(df_train, df_y_train)
