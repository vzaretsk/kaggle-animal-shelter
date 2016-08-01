import pandas as pd

# feat_file = "data/train_feat.csv"
# ext_feat_file = "data/train_ext_feat.csv"
dog_breeds_file = "data/dog_breeds_ext.csv"

# load dog breed data from American Kennel Club and create dummy variables
dog_breeds_df = pd.read_csv(dog_breeds_file)
dog_groups_df = pd.get_dummies(dog_breeds_df['Dog_group'])
dog_breeds_df = pd.concat([dog_breeds_df.drop('Dog_group', axis=1), dog_groups_df], axis=1)

for subset in ['train', 'test']:
    feat_file = "data/" + subset + "_feat.csv"
    ext_feat_file = "data/" + subset + "_ext_feat.csv"
    # load data
    feat_df = pd.read_csv(feat_file, parse_dates=['DateTime'], infer_datetime_format=True)

    # join on Breed, cat breeds will have missing values which will be dropped after
    # file is loaded
    feat_ext_df = pd.merge(left=feat_df, right=dog_breeds_df, how='left', on='Breed')
    feat_ext_df.to_csv(ext_feat_file, index=False)
