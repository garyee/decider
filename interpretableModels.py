def trainGlassBoxRegMLModel(df_train, df_y_train,model=None):
  if(model=='logistic'):
    MethodRegLog(df_train, df_y_train)
  elif(model=='linear'):
    MethodRegLinear(df_train, df_y_train)
  elif(model=='decisionTree'):
    MethodRegTree(df_train, df_y_train)
  elif(model=='GAM'):
    MethodRegGAM(df_train, df_y_train)
  else:
    MethodRegEBM(df_train, df_y_train)

def trainGlassBoxClassMLModel(df_train, df_y_train,model=None):
  if(model=='decisionTree'):
    MethodClassTree(df_train, df_y_train)
  elif(model=='rules'):
    MethodClassDecisionList(df_train, df_y_train)
  elif(model=='GAM'):
    MethodClassGAM(df_train, df_y_train)
  else:
    MethodClassEBM(df_train, df_y_train)

if(model_or_posthoc==1):
#   modelStr = executeWithWidgetValueCheck(modelStrDD,"Please select a glassbox model",'modelStr')
  if(reg_or_class==2):
    trainGlassBoxClassMLModel(X_train, y_train, modelStr)
  else:
    trainGlassBoxRegMLModel(X_train, y_train, modelStr)