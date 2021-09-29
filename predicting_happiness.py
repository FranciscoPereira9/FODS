import numpy as np
import pandas as pd
import lightgbm as ltb
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import KFold, GridSearchCV


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



# load the dataframe that was previously preprocessed
speeches_df = pd.read_csv('preprocessed_dataframe.csv')
speeches_df.dropna(subset=['Life Ladder'], inplace=True)

# create the models
lgbm_model = ltb.LGBMRegressor()

# set the list of features to use as well as the predictable
features = ['year', 'word_count', 'pos_sentiment', 'neg_sentiment', 'average_sentence_length']
predictable = ['Life Ladder']

# set the parameters that will be tested in the grid search
lgbm_grid_params = {
    'learning_rate': [0.005, 0.01],
    'n_estimators': [8, 16, 24, 28, 50],
    'num_leaves': [6, 8, 12, 16, 20, 50],  # large num_leaves helps improve accuracy but might lead to over-fitting
    'boosting_type': ['gbdt', 'dart'],  # for better accuracy -> try dart
    'max_bin': [255, 510, 1020],  # large max_bin helps improve accuracy but might slow down training progress
    'random_state': [500],
    'colsample_bytree': [0.64, 0.65],
    'subsample': [0.7, 0.75],
    'reg_alpha': [1, 1.2],
    'reg_lambda': [1, 1.2, 1.4],
}

n_splits = 5
k_fold_evaluation(speeches_df, lgbm_model, n_splits, features, predictable)

best_parameters = model_gridsearch(speeches_df, lgbm_model, n_splits, lgbm_grid_params, features, predictable)

lgbm_model_optimised = ltb.LGBMRegressor(**best_parameters)
k_fold_evaluation(speeches_df, lgbm_model_optimised, n_splits, features, predictable)



