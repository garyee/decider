from alepython import ale_plot

def applyALE(model, df_train, number_of_feature_to_display=0):
    nameOfFeature=df_train.columns.values.tolist()[number_of_feature_to_display]
    ale = ale_plot(model, df_train, nameOfFeature, monte_carlo=True)