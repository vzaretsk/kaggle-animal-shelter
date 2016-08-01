# kaggle-animal-shelter
My solution to the Kaggle shelter animal outcomes competition.

## Overview
The Kaggle shelter animal outcomes competition involves predicting the outcome of animals (adoption, etc.) at the Austin Animal Center using intake data (breed, age, etc.). The challenge details and data are available at https://www.kaggle.com/c/shelter-animal-outcomes/. Many excellent exploratory data analysis scripts provided by the community are also available at that web page.

## Model Description
I built a model based on the XGBoost classifier (https://xgboost.readthedocs.io/en/latest/) with multi-class log loss to predict the outcomes. The optimal XGB hyper-parameters were found with grid search and cross validation.

The model was split by animal type, with the cat and dog classifiers trained independently. This allowed me to monitor the individual performance on those two subsets and fine tune the hyper-parameters for each. Additionally, a split model would provide more insight on the fate of animals at the Austin Animal Center.

I used features provided in the training data and generated some additional ones. Animal breed, color, and sex were represented with one-hot encoding. In addition to the features in the data, I tested adding external features, specifically dog breed size, energy, and popularity, available on the American Kennel Club website (http://www.akc.org/). However, neither the external features nor variations on pre-processing the provided features made a large difference and all models performed in the ~0.708 log loss range.

I originally intended to blend the result of various classifiers using a second stage classifier and a hold out blending subset but that approach performed poorly. A simple arithmetic average of the XGB predicted probabilities, produced with variations of the features and hyper-parameters, gave a ~0.005 leaderboard score increase.


## File Descriptions
**clean_data, clean_data_b2** - clean the data, version b2 keeps both breeds if the animal is a mix, the original keeps only the first breed

**extract_features** - most feature engineering is done here, including creating indicator variables for name, whether the animal is of mixed breed or multi-color, and extracting date/time features such as month, hour, holiday, etc.

**external_features** - adds dog breed size, energy level, and popularity information from the American Kennel Club website, saved locally in file data/dog_breeds_ext.csv

**xgb, xgb_ext, xgb_breed2** - multi-class classification using XGBoost, includes code for grid search of optimal hyper-parameters and logging of results

**blend_xgb** - blend of the xgb predictions, done with a simple arithmetic mean

**util** - various utility functions, some are no longer used but were experimented with in other approaches
