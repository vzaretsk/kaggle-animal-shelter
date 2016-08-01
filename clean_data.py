import pandas as pd
import numpy as np
import json
from util import *

# color and breed count replacement threshold
color_threshold = 20
dog_breed_threshold = 20
cat_breed_threshold = 15

train_file = "data/train.csv"
test_file = "data/test.csv"

# load data
train_df = pd.read_csv(train_file)
train_df.rename(columns={'AnimalID': 'ID'}, inplace=True)

test_df = pd.read_csv(test_file)

# add empty outcome columns
test_df['OutcomeType'] = np.nan
test_df['OutcomeSubtype'] = np.nan

counts = train_df['AnimalType'].value_counts()
# print("number of dog and cat training examples")
# print(counts)

# combine into a single data frame
combined_df = pd.concat([train_df, test_df], axis=0)

# normalize several columns
combined_df['AnimalType'] = combined_df['AnimalType'].apply(normalize_name)
combined_df['SexuponOutcome'] = combined_df['SexuponOutcome'].apply(normalize_name)
combined_df['OutcomeType'] = combined_df['OutcomeType'].apply(normalize_name)
combined_df['OutcomeSubtype'] = combined_df['OutcomeSubtype'].apply(normalize_name)

# fill missing sex with unknown
combined_df['SexuponOutcome'].fillna(value='unknown', inplace=True)

# split color into primary color and multicolor, convert multicolor into indicator variable,
# replace low frequency colors with "rare"
combined_df[['Color', 'Multicolor']] = combined_df['Color'].str.split("/", n=1, expand=True)
combined_df.loc[~combined_df['Multicolor'].isnull(), 'Multicolor'] = 1
combined_df.loc[combined_df['Multicolor'].isnull(), 'Multicolor'] = 0

combined_df['Color'] = combined_df['Color'].apply(normalize_name)
color_counts = combined_df['Color'].value_counts()
# color_counts.to_csv("log/colors.csv")
total_colors = len(color_counts)
color_counts = color_counts[color_counts < color_threshold]
rare_colors = list(color_counts.index)
print("replacing {} colors with 'rare' out of a total of {}".format(len(rare_colors), total_colors))
print("this accounts for {} out of {} total animals".format(sum(color_counts), len(combined_df)))
combined_df['Color'].replace(to_replace=rare_colors, value="rare", inplace=True)

remaining = dict()
remaining['colors'] = combined_df['Color'].unique().tolist()

# convert name to indicator variable
combined_df.loc[~combined_df['Name'].isnull(), 'Name'] = 1
combined_df.loc[combined_df['Name'].isnull(), 'Name'] = 0

# split breed into breed and mix, mix is an indicator variable, replace low frequency
# breeds with rare
combined_df[['Breed', 'Mix']] = combined_df['Breed'].apply(clean_breed, keep_both=False)\
    .str.split("/", n=1, expand=True)

# split by animal type
dogs_df = combined_df[combined_df['AnimalType'] == 'dog']
cats_df = combined_df[combined_df['AnimalType'] == 'cat']

# dogs_df['Breed'].value_counts().to_csv("log/dog_breeds.csv")
# cats_df['Breed'].value_counts().to_csv("log/cat_breeds.csv")

dog_breeds = dogs_df['Breed'].value_counts()
total_dog_breeds = len(dog_breeds)
dog_breeds = dog_breeds[dog_breeds < dog_breed_threshold]
rare_dog_breeds = list(dog_breeds.index)
print("replacing {} dog breeds with 'rare' out of a total of {}".format(len(rare_dog_breeds), total_dog_breeds))
print("this accounts for {} out of {} total dogs".format(sum(dog_breeds), len(dogs_df)))

cat_breeds = cats_df['Breed'].value_counts()
total_cat_breeds = len(cat_breeds)
cat_breeds = cat_breeds[cat_breeds < cat_breed_threshold]
rare_cat_breeds = list(cat_breeds.index)
print("replacing {} cat breeds with 'rare' out of a total of {}".format(len(rare_cat_breeds), total_cat_breeds))
print("this accounts for {} out of {} total cats".format(sum(cat_breeds), len(cats_df)))

rare_breeds = rare_dog_breeds + rare_cat_breeds
combined_df['Breed'].replace(to_replace=rare_breeds, value="rare", inplace=True)

remaining['dog_breeds'] = combined_df.loc[combined_df['AnimalType'] == 'dog', 'Breed'].unique().tolist()
remaining['cat_breeds'] = combined_df.loc[combined_df['AnimalType'] == 'cat', 'Breed'].unique().tolist()

# convert age string into a number in months
combined_df['AgeuponOutcome'] = combined_df['AgeuponOutcome'].apply(clean_age)

# save the cleaned data
train_clean_df = combined_df[~combined_df['OutcomeType'].isnull()]
train_clean_df.to_csv("data/train_clean.csv", index=False)

test_clean_df = combined_df[combined_df['OutcomeType'].isnull()].drop(['OutcomeType', 'OutcomeSubtype'], axis=1)
test_clean_df.to_csv("data/test_clean.csv", index=False)

# save breed and color labels
with open("data/labels.json", 'w') as labels:
    json.dump(remaining, labels)
