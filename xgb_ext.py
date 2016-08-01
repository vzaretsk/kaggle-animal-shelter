import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation as cv
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from util import *

# necessary for multi-threading
if __name__ == '__main__':

    train_file = "data/train_ext_feat.csv"
    test_file = "data/test_ext_feat.csv"
    ensem_pred_file = "data/ensem_ext_xgb_pred.csv"
    test_pred_file = "data/xgb_blend/test_xgb_ext_88.csv"
    do_grid_search = False
    pred_ensem = False
    pred_test = True

    # load data
    train = load_train(train_file)

    # list of features
    # ['AgeuponOutcome', 'Breed', 'Color', 'Mix', 'Multicolor', 'Name', 'OutcomeType', 'SexuponOutcome',
    # 'Hour', 'Weekday', 'Month', 'Day', 'Workday', 'Year']

    # setup a data frame mapper, convert breed to breed mean, encode the two categorical variables,
    # pass the rest through
    # skipping standardization since it's not needed for decision tree based classifiers
    mapper = DataFrameMapper([('Breed', LabelBinarizer()),
                              ('Color', LabelBinarizer()),
                              ('SexuponOutcome', LabelBinarizer())],
                             default=None, sparse=True)

    # XGB classifier instance
    xgb = XGBClassifier(max_depth=8, learning_rate=0.02, n_estimators=500, objective='multi:softprob',
                        subsample=0.8, colsample_bytree=0.8, nthread=1)

    # pipeline used for grid search and cross validation
    pipeline = Pipeline([
        ('mapper', mapper),
        ('xgb', xgb)])

    # grid search parameters
    param_grid = {'xgb__max_depth': [10],
                  'xgb__learning_rate': [0.02],
                  'xgb__n_estimators': [500],
                  'xgb__subsample': [0.8],
                  'xgb__colsample_bytree': [0.8]}

    # data set to use for grid search
    data_set = 'cat'
    X, y = train[data_set]

    # cross validation strategy
    skf = cv.StratifiedKFold(y, n_folds=10, shuffle=True)

    # grid search, verbose 0 is minimal output, 1 is some output, 2 is detailed output
    gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='log_loss', cv=skf,
                      n_jobs=5, refit=False, verbose=1)

    # comment added to the first line of the log file
    note_str = "{} subset using XGB classifier, with external features, binarize breed, color, and sex".format(data_set)

    if do_grid_search:
        gs.fit(X, y)
        print("best score: {s:.4f}".format(s=-gs.best_score_))
        write_search_log(gs, note_str)

    # found by grid search, binarize breed, color, and sex
    dog_params = {'xgb__colsample_bytree': 0.8, 'xgb__n_estimators': 500, 'xgb__learning_rate': 0.02,
                  'xgb__subsample': 0.8, 'xgb__max_depth': 10, 'xgb__nthread': 5}
    cat_params = {'xgb__colsample_bytree': 0.8, 'xgb__n_estimators': 500, 'xgb__learning_rate': 0.02,
                  'xgb__subsample': 0.8, 'xgb__max_depth': 10, 'xgb__nthread': 5}

    # train pipeline on both animal types
    param_dict = {'cat': cat_params, 'dog': dog_params}
    fitted = fit_animals(pipeline, param_dict, train)

    # load unused data
    ensem = load_train(train_file, subset='ensem')

    ensem_pred_lst = []
    for animal, clf in fitted.items():
        X, y = train[animal]
        y_pred = clf.predict_proba(X)
        print("log loss on the training {a:} subset {l:.4f}".format(a=animal, l=log_loss(y, y_pred)))

        X, y = ensem[animal]
        y_pred = clf.predict_proba(X)
        print("log loss on the ensemble {a:} subset {l:.4f}".format(a=animal, l=log_loss(y, y_pred)))

        ensem_df = pd.DataFrame(index=y.index)
        ensem_df['animal'] = animal
        ensem_df['outcome'] = y
        ensem_df = ensem_df.reindex(columns=list(ensem_df.columns)+list(clf.steps[1][1].classes_))
        ensem_df[list(clf.steps[1][1].classes_)] = y_pred
        ensem_pred_lst.append(ensem_df)

    if pred_ensem:
        print("saving ensemble subset predictions")
        ensem_pred_df = pd.concat(ensem_pred_lst, axis=0)
        ensem_pred_df.sort_index(inplace=True)
        ensem_pred_df.to_csv(ensem_pred_file)

    if pred_test:
        print("training on the full set and saving test predictions")
        # load test data and full train data
        test = load_test(test_file)
        full = load_train(train_file, subset='all')
        full_fit = fit_animals(pipeline, param_dict, full)

        test_pred_lst = []
        for animal, clf in fitted.items():
            X, id_df = test[animal]
            y_pred = clf.predict_proba(X)
            test_df = id_df.reindex(columns=list(id_df.columns)+list(clf.steps[1][1].classes_))
            test_df[list(clf.steps[1][1].classes_)] = y_pred
            test_pred_lst.append(test_df)

        test_df = pd.concat(test_pred_lst, axis=0)
        test_df.sort_index(inplace=True)
        test_df.to_csv(test_pred_file, index=False)
