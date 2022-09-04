from xgboost import XGBRegressor, XGBClassifier
# from autosklearn.experimental.askl2 import AutoSklearn2Classifier
# from autosklearn.regression import AutoSklearnRegressor
from sklearn.metrics import accuracy_score, r2_score

def trainBlackBoxModel(reg_or_class,df_train, df_y_train,df_test,df_y_test,model=None):
    if(reg_or_class==2):
      return trainBlackBoxClassModel(df_train, df_y_train,df_test,df_y_test,model)
    else:
      return trainBlackBoxRegModel(df_train, df_y_train,df_test,df_y_test,model)


def trainBlackBoxClassModel(df_train, df_y_train,df_test,df_y_test,model=None):
    global seed
    my_model= None
    # if(model=='autoML'):
        # my_model = AutoSklearn2Classifier(random_state=seed)
    # else:
    my_model = XGBClassifier(random_state=seed)

    # my_model.fit(df_train.values, df_y_train)
    my_model.fit(df_train, df_y_train)
    test_predictions = my_model.predict(df_test)
    print("Accuracy score", accuracy_score(df_y_test, test_predictions))
    return my_model

def trainBlackBoxRegModel(df_train, df_y_train,df_test,df_y_test,model=None):
    global seed
    my_model= None
    # if(model=='autoML'):
    #   my_model = AutoSklearnRegressor(
    #   time_left_for_this_task=120,
    #   per_run_time_limit=30,
    #   tmp_folder='/tmp/autosklearn_regression_example_tmp',
    # )
    # else:
    my_model = XGBRegressor(n_estimators=1000,verbose=False)
    
    my_model.fit(df_train, df_y_train)
    # my_model.fit(df_train.values, df_y_train)
    test_predictions = my_model.predict(df_test)
    print("Test R2 score:", r2_score(df_y_test, test_predictions))
    return my_model