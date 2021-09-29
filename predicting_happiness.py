import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV, train_test_split


def k_fold_evaluation(speeches_df, model, n_splits, features, predictable):
    kf = KFold(n_splits=n_splits, shuffle=True)

    i = 0

    r2_scores = []
    mse_scores = []

    for train, test in kf.split(speeches_df):
        X_train = speeches_df.iloc[train][features]
        y_train = speeches_df.iloc[train][predictable]

        X_test = speeches_df.iloc[test][features]
        y_test = speeches_df.iloc[test][predictable]

        model.fit(X_train, y_train)

        expected_y = y_test
        predicted_y = lgbm_model.predict(X_test)

        r2_scores.append(metrics.r2_score(expected_y, predicted_y))
        mse_scores.append(metrics.mean_squared_error(expected_y, predicted_y))

        i += 1

    print(r'The average r2 score +- std: {} +- {}'.format(np.mean(r2_scores), np.std(r2_scores)))
    print(r'The average mse score +- std: {} +- {}'.format(np.mean(mse_scores), np.std(mse_scores)))



def model_gridsearch(speeches_df, model, n_splits, gridParams, features, predictable):
    grid = GridSearchCV(model, gridParams, verbose=1, cv=n_splits, n_jobs=-1)

    grid.fit(speeches_df[features], speeches_df[predictable])
    print('Best parameters: ', grid.best_params_)

    return grid.best_params_

def test_set_evaluation(speeches_df_train, speeches_df_test, model_test, features, predictable):

    # we split the training set so that we can prevent overfitting of the model
    speeches_df_train, speeches_df_validation = train_test_split(speeches_df_train, test_size=0.2, shuffle=True,
                                                                 random_state=42)

    model_test.fit(speeches_df_train[features], speeches_df_train[predictable],
                   eval_set=[(speeches_df_validation[features], speeches_df_validation[predictable]),
                             (speeches_df_train[features], speeches_df_train[predictable])],
                   early_stopping_rounds=10, verbose=5)


    expected_y = speeches_df_test[predictable]
    predicted_y = lgbm_model.predict(speeches_df_test[features])


    print('Plot metrics during training...')
    ax = lgb.plot_metric(model_test.evals_result_, metric='l2')
    plt.show()

    ax = lgb.plot_importance(model_test, max_num_features=10)
    plt.show()


    print('The R2 score on the test set: ', metrics.r2_score(expected_y, predicted_y))
    print('The MSE score on the test set: ', metrics.mean_squared_error(expected_y, predicted_y))


# load the dataframe that was previously preprocessed
speeches_df = pd.read_csv('preprocessed_dataframe.csv')
speeches_df.dropna(subset=['Life Ladder'], inplace=True)

speeches_df_train, speeches_df_test = train_test_split(speeches_df, test_size=0.2, shuffle=True)


# set the list of features to use as well as the predictable
features = ['year', 'word_count', 'pos_sentiment', 'neg_sentiment', 'average_sentence_length', "Log GDP per capita", 'Social support', 'Freedom to make '
                 'life choices', 'Generosity', 'Perceptions of corruption']
predictable = ['Life Ladder']

# set the parameters that will be tested in the grid search
lgbm_grid_params = {
    'learning_rate': [0.005, 0.01],
    'n_estimators': [16, 24, 28, 1000],
    'num_leaves': [12, 16, 20],  # large num_leaves helps improve accuracy but might lead to over-fitting
    'boosting_type': ['gbdt'],  # for better accuracy -> try dart
    'max_bin': [255, 510, 1020],  # large max_bin helps improve accuracy but might slow down training progress
    'random_state': [500],
    'colsample_bytree': [0.64, 0.65],
    'subsample': [0.7, 0.75],
    'reg_alpha': [1, 1.2],
    'reg_lambda': [1, 1.2, 1.4],
}


# create the models
lgbm_model = lgb.LGBMRegressor()
n_splits = 5

# evaluation before optimisation
k_fold_evaluation(speeches_df_train, lgbm_model, n_splits, features, predictable)

test_set_evaluation(speeches_df_train, speeches_df_test, lgbm_model, features, predictable)

# grid search for the best parameters
# best_parameters = model_gridsearch(speeches_df_train, lgbm_model, n_splits, lgbm_grid_params, features, predictable)
best_parameters =  {'boosting_type': 'gbdt', 'colsample_bytree': 0.65, 'learning_rate': 0.01, 'max_bin': 1020,
                   'n_estimators': 1000, 'num_leaves': 20, 'random_state': 500, 'reg_alpha': 1, 'reg_lambda': 1.4, 'subsample': 0.7}


# look at increase of scores
lgbm_model_optimised = lgb.LGBMRegressor(**best_parameters)
k_fold_evaluation(speeches_df_train, lgbm_model_optimised, n_splits, features, predictable)

# final evalutation on unseen test set
lgbm_model_optimised_test = lgb.LGBMRegressor(**best_parameters)
test_set_evaluation(speeches_df_train, speeches_df_test, lgbm_model_optimised_test, features, predictable)



