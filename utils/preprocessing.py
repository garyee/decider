from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler,EditedNearestNeighbours,TomekLinks
import pandas as pd
import numpy as np

from utils.enums import TaskType
from utils.helper import getCategoricalColsNameList
from utils.columtransformer import get_feature_names

def preprocess(X_train, X_test, y_train, y_test,taskType):
  # display_missing_values_table_chart(X_train)
  delete_missing_values_columns(X_train,X_test)
  
  # display_high_cardinalitity(X_train)
  delete_high_cardinalitity(X_train,X_test)
  
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    X_train,X_test=impute(X_train,X_test)
  #X_train.head()

  X_train,X_test=encodeCategoricalCols(X_train,X_test)
  # #X_train.head()

  if taskType==TaskType.CLASSIFICATION:
    showBalance(y_train)
    if getRatio(y_train)>=2:
      X_train,y_train = resample(X_train,y_train,strategy=1)

  return [X_train, X_test, y_train, y_test]

# Delete cols with to much missing
def display_missing_values_table_chart(df,axis=1):
    percent_missing = df.isnull().sum() * 100 / len(df)
    print(percent_missing)
    #percent_missing.plot.bar()

def delete_missing_values_columns(df_train,df_test=None,nanThreshold=65):
    cols_with_nan = [cname for cname in df_train.columns if 100 * df_train[cname].isnull().sum()/ len(df_train[cname]) > nanThreshold]
    if (len(cols_with_nan)>0):
      df_train.drop(cols_with_nan,axis='columns', inplace=True)
      print('Deleted Columns: ',cols_with_nan,'because it/they had more than',nanThreshold,'% of null values')
    if(df_test is not None):
      df_test.drop(cols_with_nan,axis='columns', inplace=True)

#@title Delete high cardinal categorical cols
def get_cardinality_percent(df):
  res = pd.Series(dtype='int')
  #catCols,_ = getCategoricalColsNameList(df)
  for colName in df.columns:
    col_cardinality=len(pd.unique(df[colName]))
    percentage=100/len(df[colName])*col_cardinality
    res=pd.concat([res,pd.Series([percentage],index =[colName])])
  return res
  #df.apply(pd.Series.nunique)

def display_high_cardinalitity(df,axis=1):
    percent_cardinalitity = get_cardinality_percent(df)
    percent_cardinalitity.plot.bar()

def delete_high_cardinalitity(df_train,df_test=None,cardinality_threshold=50):
  percent_cardinalitity = get_cardinality_percent(df_train)
  highCardinalCols=percent_cardinalitity[lambda x: x>cardinality_threshold]
  highCardinalColsIndexList=highCardinalCols.index.values.tolist()
  if(len(highCardinalColsIndexList)):
    df_train.drop(highCardinalColsIndexList,axis=1,inplace=True)
    if(df_test is not None):
      df_test.drop(highCardinalColsIndexList,axis=1,inplace=True)
    print("The column(s) '",highCardinalColsIndexList,"' is/are droped as it was a high cardinality feature")

#@title Impute all cols with missing values
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings

def impute(df_train,df_test=None,):
  cat_cols,num_cols=getCategoricalColsNameList(df_train)
  column_trans = ColumnTransformer(
  [("num",SimpleImputer(strategy='mean'),num_cols),
  ("cat", SimpleImputer(strategy='constant'), cat_cols)],
  remainder='passthrough')
  missingdata_df = df_train.columns[df_train.isnull().any()].tolist()
  
  #print('Count of Null\'s in the dataframe: ',len(missingdata_df))
  df_train_imputed = pd.DataFrame(column_trans.fit_transform(df_train))
  
  #bring columns and dtypes back
  originalColNames = get_feature_names(column_trans)
  df_train_imputed.columns = originalColNames
  df_train_imputed=df_train_imputed.astype(df_train.dtypes.to_dict())

  df_test_imputed=None
  if(df_test is not None):
      df_test_imputed = pd.DataFrame(column_trans.transform(df_test))
      df_test_imputed.columns = originalColNames
      df_test_imputed=df_test_imputed.astype(df_test.dtypes.to_dict())
  if(missingdata_df):
      print('The following columns have been imputed:',','.join(missingdata_df))
  return df_train_imputed,df_test_imputed

def encodeCategoricalCols(df_train,df_test=None):
  cat_cols,num_cols=getCategoricalColsNameList(df_train)
  res_test=df_test
  res_train=df_train
  if (cat_cols is not None and isinstance(cat_cols,list) and len(cat_cols)>0):
    OH_encoder = OneHotEncoder(handle_unknown='error',drop='if_binary',sparse=False)
    df_train_OH_cols = pd.DataFrame(OH_encoder.fit_transform(df_train[cat_cols]))
    df_train_OH_cols.columns = OH_encoder.get_feature_names(cat_cols)
    df_train.drop(cat_cols, axis=1, inplace=True)
    res_train = pd.concat([df_train, df_train_OH_cols], axis=1)
    if(df_test is not None):
        df_test_OH_cols = pd.DataFrame(OH_encoder.transform(df_test[cat_cols]))
        df_test_OH_cols.columns = OH_encoder.get_feature_names(cat_cols)
        df_test_OH_cols.index = df_test.index
        df_test.drop(cat_cols, axis=1, inplace=True)
        res_test = pd.concat([df_test, df_test_OH_cols], axis=1)
    print('Encoded the cols:'+', '.join(cat_cols)  )
  return res_train,res_test

#@title Check and counter class imbalance if ratio > 1:2
def getCountsPerClass(df):
  return pd.value_counts(df,sort=True)

def getRatio(df):
  count_classes=getCountsPerClass(df)
  return count_classes[0]/count_classes[1]

def printClassCount(count_classes):
    print(count_classes)

def showBalance(df):
    count_classes=getCountsPerClass(df)
    printClassCount(count_classes)
    print('1 :',count_classes[0]/count_classes[1])
    #count_classes.plot.bar(rot=0)
    
def resample(X,y,strategy='auto',algo=None,random_state=1):
    my_resampler= None
    if(algo=='SMOTETomek'):
        my_resampler = SMOTETomek(random_state=random_state)
    elif(algo=='Tomek'):
        my_resampler = TomekLinks(random_state=random_state)
    elif(algo=='ENN'):
        my_resampler = EditedNearestNeighbours(random_state=random_state)
    elif(algo=='SMOTE'):
        my_resampler = SMOTE(random_state=random_state,sampling_strategy = strategy)
    else:
        my_resampler = RandomUnderSampler(random_state=random_state,sampling_strategy = strategy,)
    
    X_resampled, y_resampled = my_resampler.fit_resample(X, y)
    X_resampled = pd.DataFrame(X_resampled)
    X_resampled.columns = X.columns
    y_resampled = pd.Series(y_resampled)
    printClassCount(y_resampled,'resampled by '+ (algo or 'RUS')+'\n')
    return X_resampled,y_resampled

# if reg_or_class==2:
#   showBalance(y_train)
#   if getRatio(y_train)>=2:
#     X_train,y_train = resample(X_train,y_train,strategy=1)