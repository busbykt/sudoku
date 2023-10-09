'''

'''

# %%
# dependencies
import pandas as pd
import numpy as np
import sudokum as su
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config

set_config(transform_output="pandas")

# %%
categories = ['1','2','3','4','5','6','7','8','9']
predictors = categories[:-1]
def generateSingleLineSudoku(samples):

    df_list = []
    for i in np.arange(0,samples):
        df_list.append(
            np.random.choice(
                np.arange(1,10,1),size=9,replace=False
            )
        )

    return pd.DataFrame(
        df_list, 
        columns=predictors+['target']
    )
# %%
train_df = generateSingleLineSudoku(10000)
test_df = generateSingleLineSudoku(10000)

# %%
ohe = OneHotEncoder(
    sparse_output=False,
    categories=[categories]*train_df.shape[1]
)
ohe_train_df = ohe.fit_transform(train_df)
ohe_test_df = ohe.transform(test_df)

# %%
y = ohe_train_df.filter(like='target').values
x = ohe_train_df[[x for x in ohe_train_df if 'target' not in x]].values

y_val = ohe_test_df.filter(like='target').values
x_val = ohe_test_df[[x for x in ohe_test_df if 'target' not in x]].values
# %%
# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(72,), name='puzzle'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(9, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
# %%
model.fit(x,y, epochs=5, validation_data=(x_val[:-100],y_val[:-100]))
# %%
y_pred = model.predict(x_val[-100:]).round(0)
y_pred = pd.DataFrame(y_pred,columns=categories)
y_true = pd.DataFrame(y_val[-100:], columns=categories)
# %%
