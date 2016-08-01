import pandas as pd
import numpy as np
from pandas.tseries.holiday import Holiday, HolidayCalendarFactory, USFederalHolidayCalendar, FR
from sklearn.cross_validation import train_test_split
from util import *

# train_file = "data/train_clean.csv"
# test_file = "data/test_clean.csv"
# out_train_file = "data/train_feat.csv"
# out_test_file = "data/test_feat.csv"

train_file = "data/train_clean_v2.csv"
test_file = "data/test_clean_v2.csv"
out_train_file = "data/train_feat_v2.csv"
out_test_file = "data/test_feat_v2.csv"

# load data
train_df = pd.read_csv(train_file, parse_dates=['DateTime'], infer_datetime_format=True)
test_df = pd.read_csv(test_file, parse_dates=['DateTime'], infer_datetime_format=True)

# add empty outcome columns
test_df['OutcomeType'] = np.nan
test_df['OutcomeSubtype'] = np.nan

# combine into a single data frame
combined_df = pd.concat([train_df, test_df], axis=0)

# create features for the hour of the day, day of the week,
# month, and day of the month from the DateTime information
combined_df['Hour'] = combined_df['DateTime'].dt.hour + combined_df['DateTime'].dt.minute/60
combined_df['Weekday'] = combined_df['DateTime'].dt.weekday
combined_df['Month'] = combined_df['DateTime'].dt.month
combined_df['Day'] = combined_df['DateTime'].dt.day
combined_df['Year'] = combined_df['DateTime'].dt.year

# # print(combined_df['Hour'].describe())
# import matplotlib.pyplot as plt
# # leaving out .values causes problems
# plt.hist(combined_df['Hour'].values, bins=48, range=(-0.5, 24.5))
# combined_df['Weekday'].hist(bins=7)
# plt.show()


# create an indicator variable for business days, first need to create a calendar of holidays
# define the Friday after Thanksgiving as a holiday and add it to USFederalHolidayCalendar
# no need to modify calendar start and end dates as they 1970-01-01 and 2030-12-31 by default
ThanksFri = Holiday("Thanksgiving Friday", month=11, day=1, offset=pd.DateOffset(weekday=FR(4)))
Holidays = HolidayCalendarFactory('Holidays', USFederalHolidayCalendar(), ThanksFri)

# earliest and latest dates present in the data
start_date = combined_df['DateTime'].min().date()
end_date = combined_df['DateTime'].max().date()

# index of holidays in the date range
hol_index = Holidays().holidays(start=start_date, end=end_date, return_name=False)

# create indicator variable for workday
combined_df['Workday'] = combined_df['DateTime'].apply(is_workday, args=(hol_index,))


# save the data with new features
test_feat_df = combined_df[combined_df['OutcomeType'].isnull()].drop(['OutcomeType', 'OutcomeSubtype'], axis=1)
test_feat_df.to_csv(out_test_file, index=False)

# split the training data set into a training set and an ensembling set,
# perform a stratified split by animal type, record the set type in column Set
strat = combined_df.loc[~combined_df['OutcomeType'].isnull(), 'AnimalType'] == "cat"
train_df, ensem_df = train_test_split(combined_df[~combined_df['OutcomeType'].isnull()],
                                                train_size=0.9, stratify=strat)
combined_df.loc[train_df.index, 'Set'] = "train"
combined_df.loc[ensem_df.index, 'Set'] = "ensem"

combined_df[~combined_df['OutcomeType'].isnull()].to_csv(out_train_file, index=False)
