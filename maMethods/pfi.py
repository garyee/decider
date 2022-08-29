from sklearn.inspection import permutation_importance

def applyPFI(model,df_train,df_train_y):
    result = permutation_importance(model,
                                    df_train,
                                    df_train_y,
                                    n_repeats=10,
                                    random_state=42,
                                    n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    fig, ax = plt.subplots()
    fpi=ax.boxplot(result.importances[sorted_idx].T,vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importances Titanic dataset (training set)")
    fig.tight_layout()
    plt.show()