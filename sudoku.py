'''

'''

# %%
# dependencies
import pandas as pd
import numpy as np
import sudokum as su

# %%
def generateSingleLineSudoku(samples):

    df_list = []
    for i in np.arange(0,samples):
        df_list.append(
            np.random.choice(
                np.arange(1,10,1),size=9,replace=False
            )
        )

    return pd.DataFrame(df_list, columns=np.arange(1,9,1).tolist()+['target'])
# %%
train_df = generateSingleLineSudoku(10000)
test_df = generateSingleLineSudoku(10000)

# %%
