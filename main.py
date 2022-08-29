# import os

# def restart_runtime():
#   os.kill(os.getpid(), 9)

# restart_runtime()

initialConfig={
    targetLabel:'survived',
    regression:false,
    classification:false,
    glob_or_local:1,
    model_or_posthoc:2,
}

#display_missing_values_table_chart(X_train)
delete_missing_values_columns(X_train,X_test)

#display_high_cardinalitity(X_train)
delete_high_cardinalitity(X_train,X_test)

with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  X_train,X_test=impute(X_train,X_test)

X_train,X_test=encodeCategoricalCols(X_train,X_test)

printCorrMatrix(X_train)