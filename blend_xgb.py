import pandas as pd
import numpy as np
from glob import glob

data_dir = "data/xgb_blend/"
blend_pred_file = "data/xgb_blend/test_blend_xgb.csv"
pred_col_lst = ['adoption', 'died', 'euthanasia', 'return_to_owner', 'transfer']

# load individual predictions
first = True
test_lst = []
for file in glob(data_dir + "test_xgb_*.csv"):
    if first:
        test_df = pd.read_csv(file, usecols=['ID'] + pred_col_lst)
        blend_df = test_df.copy()
        first = False
    else:
        test_df = pd.read_csv(file, usecols=pred_col_lst)

    test_lst.append(test_df[pred_col_lst].values)

blend_arr = np.mean(test_lst, axis=0)
blend_df.loc[:, pred_col_lst] = blend_arr

blend_df.to_csv(blend_pred_file, index=False)
