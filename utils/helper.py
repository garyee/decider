#@title Helper Functions (read/write files etc)
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

tmpConfig=None
seed=42

def uplodedFileToDf(filesDict,index,save=False):
  fileContent=list(filesDict.values())[index]
  if save==True:
    with open("./tmp_save"+str(index)+".csv", "wb") as fp:
      fp.write(fileContent)
  content = io.StringIO(fileContent.decode('utf-8'))
  return pd.read_csv(content)

def readUploadedFilesToDfs(uploader,save=False):
  result=()
  for x in range(len(uploader)):
    result= result+(uplodedFileToDf(uploader,x,save),)
  return result

def readUploadedFilesToDfsAndSave(uploader):
  return readUploadedFilesToDfs(uploader,True)

# def checkForTempFiles():
#   X_train=None
#   X_test=None
#   if os.path.isfile("./tmp_save0.csv"):
#       X_train=pd.read_csv("./tmp_save0.csv")
#       if os.path.isfile("./tmp_save1.csv"):
#         X_test=pd.read_csv("./tmp_save1.csv")
#   return X_train,X_test

# def getTmpConfigProperty(property):
#   global tmpConfig
#   if tmpConfig is None and os.path.isfile("./config.json"):
#     with open('./config.json', 'r') as f:
#       tmpConfig = json.load(f)
#   if tmpConfig is None:
#     return None
#   return tmpConfig.get(property, None)

# def setTmpConfigProperty(property,value):
#   global tmpConfig
#   if os.path.isfile("./config.json"):
#     with open('./config.json', 'r') as f:
#       tmpConfig = json.load(f)
#   if(tmpConfig is None):
#     tmpConfig={}
#   tmpConfig[property]=value
#   with open('./config.json', 'w') as f:
#     json.dump(tmpConfig, f)

# def getValueFromWidget(widget):
#   return widget.value

# def executeWithWidgetValueCheck(widget,msg,saveWithProperty=None,cb=getValueFromWidget):
#   if widget is not None and ((isinstance(widget.value, dict) and len(widget.value)> 0) or (not isinstance(widget.value, dict) and widget.value is not None)):
#     if saveWithProperty is not None:
#       setTmpConfigProperty(saveWithProperty,widget.value)
#     return cb(widget)
#   else:
#     raise RuntimeError(msg)

def getTrainAndTestSetXandY(X_train,X_test,y_label):
  y_train=X_train[y_label]
  X_train.drop([y_label],axis=1,inplace=True)
  if (X_test is not None and type(X_test) is not list) and y_label not in X_test:
    raise RuntimeError('The test set has no target variable column named: '+y_label+' !')
  if X_test is not None and type(X_test) is not list:
    y_test=X_test[y_label]
    X_test.drop([y_label],axis=1,inplace=True)
    return X_train,X_test,y_train,y_test
  return train_test_split(X_train, y_train, test_size=0.3, random_state=42)

def getCategoricalAndNummericalColsNameList(df):
  cols = list(df.columns.values)
  num_cols = list(df._get_numeric_data().columns.values)
  return list(set(cols) - set(num_cols)),num_cols

def getCatlAndNumColsCount(df):
  total = len(list(df.columns.values))
  cat_cols,num_cols = getCategoricalAndNummericalColsNameList(df)
  return total,len(cat_cols),len(num_cols)
