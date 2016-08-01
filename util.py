import numpy as np
import pandas as pd
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


# normalizes a string by converting to lower case, replacing all spaces with _
def normalize_name(s: str):
    if s is np.nan or s is None:
        return np.nan
    s = s.strip().lower()
    return re.sub(r"\s+", "_", s)


# cleans the breed type, returning a normalized form of the primary breed and an indicator
# variable if it's a mix
# returning a pd.Series is very slow for some reason
# instead return a single string with / as the separator
def clean_breed(s: str, keep_both: bool = True):
    s = s.lower()
    if "/" in s:
        b, b2, *_ = s.split("/")
        b = normalize_name(b)
        b2 = normalize_name(b2)
        m = "1"
    elif "mix" in s:
        b = s.replace("mix", "")
        b, b2, m = normalize_name(b), "", "1"
    else:
        b, b2, m = normalize_name(s), "", "0"

    if keep_both:
        return "/".join([b, b2, m])
    else:
        return "/".join([b, m])


# converts age string into an age number in months
# returns 1000 if age information is missing
# i'm assuming that lack of age information is greater than any other age,
# i.e. old animals and animals with age information have similar properties
def clean_age(a: str):
    if a is np.nan:
        return 1000
    age, unit = a.strip().lower().split()
    age = float(age)
    if "week" in unit:
        age *= 12/52
    elif "year" in unit:
        age *= 12
    return age


# return 0 if the date is a weekend or holiday
# holidays is an index of holiday dates
def is_workday(date: datetime, holidays: pd.DatetimeIndex):
    if date in holidays or date.weekday() > 4:
        return 0
    else:
        return 1


# derivative class of the sklearn LabelEncoder, transform is modified to return a value for unseen labels,
# using the nearest alphabetically sorted label as the proxy
# this is needed in case the bootstrap sample in a random forest is missing all examples of a low occurrence breed
class ImputeLabelEncoder(LabelEncoder):
    def transform(self, y):
        check_is_fitted(self, 'classes_')

        # classes = np.unique(y)
        # _check_numpy_unicode_bug(classes)
        # if len(np.intersect1d(classes, self.classes_)) < len(classes):
        #     diff = np.setdiff1d(classes, self.classes_)
        #     raise ValueError("y contains new labels: %s" % str(diff))
        return np.searchsorted(self.classes_, y)


# loads the train csv file, splits by animal type, and returns a dict of X and y tuples
def load_train(train_file: str, subset: str='train'):
    # load data
    train_df = pd.read_csv(train_file, parse_dates=['DateTime'], infer_datetime_format=True)

    # select a subtset of the training data
    if subset == 'train':
        train_df = train_df[train_df['Set'] == subset].copy()
        train_df.drop('Set', axis=1, inplace=True)
    elif subset == 'ensem':
        train_df = train_df[train_df['Set'] == subset].copy()
        train_df.drop('Set', axis=1, inplace=True)
    elif subset == 'all':
        train_df.drop('Set', axis=1, inplace=True)
    else:
        raise ValueError("invalid set type {}".format(subset))

    # drop unneeded columns
    train_df.drop(['DateTime', 'ID', 'OutcomeSubtype'], axis=1, inplace=True)

    # split by animal type and then drop AnimalType
    cat_df = train_df[train_df['AnimalType'] == "cat"].drop('AnimalType', axis=1).copy()
    dog_df = train_df[train_df['AnimalType'] != "cat"].drop('AnimalType', axis=1).copy()

    # list of features
    # ['AgeuponOutcome', 'Breed', 'Color', 'Mix', 'Multicolor', 'Name', 'OutcomeType', 'SexuponOutcome',
    # 'Hour', 'Weekday', 'Month', 'Day', 'Workday', 'Year']

    # drop empty external dog feature columns
    ext_dog_col = ['Energy', 'Size', 'Popularity', 'herding', 'hound', 'misc', 'non_sporting',
                   'sporting', 'terrier', 'toy', 'working']
    if 'Energy' in cat_df.columns:
        cat_df.drop(ext_dog_col, axis=1, inplace=True)

    train = dict()
    train['cat'] = (cat_df.drop('OutcomeType', axis=1), cat_df['OutcomeType'])
    train['dog'] = (dog_df.drop('OutcomeType', axis=1), dog_df['OutcomeType'])

    return train


# given an input X of animals of various breeds and their outcome y, output the mean
# for each outcome type per breed
# expects both X and y as arrays of labels
class BreedMean(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        # get_dummies and combine in one data frame
        y = np.atleast_1d(y)
        dummies_df = pd.get_dummies(y.flatten())
        self.classes_ = list(dummies_df.columns)
        joined_df = pd.concat([pd.DataFrame({'breed': X.flatten()}), dummies_df], axis=1)

        breed_gp = joined_df.groupby('breed')
        self.class_means_ = dict()
        for breed, group in breed_gp:
            self.class_means_[breed] = group.mean().values

        self.grand_mean_ = dummies_df.mean().values

        return self

    def transform(self, X):
        means_arr = np.zeros(shape=(X.shape[0], len(self.classes_)), dtype=float)

        # look up the value of each breed and return it's mean outcome
        # if a new breed is encountered, return the grand mean
        for row, breed in enumerate(X.flatten()):
            try:
                means_arr[row, :] = self.class_means_[breed]
            except KeyError:
                means_arr[row, :] = self.grand_mean_

        return means_arr


# write a log file of grid search results
def write_search_log(gs: GridSearchCV, note: str=""):
    log_file = "log/grid_search_{time:}.csv".format(time=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    with open(log_file, 'w') as log:
        print("\"{n:}\",,".format(n=note), file=log)
        print("score_mean,score_std,params", file=log)
        for params, score_mean, scores in gs.grid_scores_:
            score_std = np.std(scores)
            params = repr(params)
            print("{m:.4f},{s:.4f},\"{p:}\"".format(m=-score_mean, s=score_std, p=params), file=log)

    return


# loads the test csv file, splits by animal type, and returns a dict of X and id tuples
def load_test(test_file: str):
    # load data
    test_df = pd.read_csv(test_file, parse_dates=['DateTime'], infer_datetime_format=True)

    # drop unneeded columns
    test_df.drop(['DateTime'], axis=1, inplace=True)

    # split by animal type and then drop AnimalType
    cat_df = test_df[test_df['AnimalType'] == "cat"].drop('AnimalType', axis=1).copy()
    dog_df = test_df[test_df['AnimalType'] != "cat"].drop('AnimalType', axis=1).copy()

    # list of features
    # ['AgeuponOutcome', 'Breed', 'Color', 'Mix', 'Multicolor', 'Name', 'OutcomeType', 'SexuponOutcome',
    # 'Hour', 'Weekday', 'Month', 'Day', 'Workday', 'Year']

    # drop empty external dog feature columns
    ext_dog_col = ['Energy', 'Size', 'Popularity', 'herding', 'hound', 'misc', 'non_sporting',
                   'sporting', 'terrier', 'toy', 'working']
    if 'Energy' in cat_df.columns:
        cat_df.drop(ext_dog_col, axis=1, inplace=True)

    train = dict()
    train['cat'] = (cat_df.drop('ID', axis=1), cat_df[['ID']])
    train['dog'] = (dog_df.drop('ID', axis=1), dog_df[['ID']])

    return train


# train the provided classifier on both data sets in the data dict using the provided parameters
def fit_animals(pipeline: Pipeline, param_dict: dict, data_dict: dict):
    fitted = dict()
    for animal, params in param_dict.items():
        clf = clone(pipeline)
        clf.set_params(**params)
        clf.fit(*data_dict[animal])
        fitted[animal] = clf
    return fitted


# converts a multi-class np.array into a list of dicts with one-hot encoded features
def array_to_dict(class_arr: np.array):
    result = list()
    for row in class_arr:
        # create a dictionary for each row
        feats = dict()
        for c in row:
            # assumes feature value is either a string or np.nan
            # np.nan can't be used on strings so using alternative method to detect np.nan
            if isinstance(c, str):
                feats[c] = 1
        result.append(feats)
    return result
