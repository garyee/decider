def globalMethod(model,df_train,df_test,df_train_y,df_test_y):
    applyGlobalSurrogate(model, df_train,df_test)
    #applyPFI(model,df_train,df_train_y)
    #applyALE(model,df_train,)
    #applyICE(model,df_train,df_test)
    #applySHAP(model,df_train,df_test)

def localMethod(model,df_train,df_test,df_train_y):
    applyLime(model,df_train,df_test,df_train_y)

if(model_or_posthoc==2):
  model=trainBlackBoxModel(reg_or_class,X_train, y_train,X_test,y_test)
  if(glob_or_local==2):
    localMethod(model,X_train,X_test,y_train)
  else:
    globalMethod(model,X_train,X_test,y_train,y_test)