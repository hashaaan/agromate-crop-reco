import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
# %matplotlib inline

import tensorflow.keras as tf
from sklearn.model_selection import train_test_split

# load data sets
ratings_df = pd.read_csv("crop-data/ratings.csv") 
crops_df = pd.read_csv("crop-data/crops.csv")

ratings_df.head()
crops_df.head()

# print(ratings_df.shape)
# print(ratings_df.user_id.nunique())
# print(ratings_df.crop_id.nunique())
# print(ratings_df.isna().sum())

Xtrain, Xtest = train_test_split(ratings_df, test_size=0.2, random_state=1)

# print(f"Shape of train data: {Xtrain.shape}")
# print(f"Shape of test data: {Xtest.shape}")

# get the number of unique entities in crops and users columns
ncrop_id = ratings_df.crop_id.nunique()
nuser_id = ratings_df.user_id.nunique()

# define vocabulary size and input length
vocab_size = 550
input_len = 4

# crop input network
input_crops = tf.layers.Input(shape=[1])
embed_crops = tf.layers.Embedding(vocab_size,input_len)(input_crops)
crops_out = tf.layers.Flatten()(embed_crops)

# user input network
input_users = tf.layers.Input(shape=[1])
embed_users = tf.layers.Embedding(vocab_size,input_len)(input_users)
users_out = tf.layers.Flatten()(embed_users)

conc_layer = tf.layers.Concatenate()([crops_out, users_out])
x = tf.layers.Dense(128, activation='relu')(conc_layer)
x_out = x = tf.layers.Dense(1, activation='relu')(x)
model = tf.Model([input_crops, input_users], x_out)

# set optimizer
opt = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error')

# show model summery
model.summary()

hist = model.fit([Xtrain.crop_id, Xtrain.user_id], Xtrain.rating, 
                 batch_size=64, 
                 epochs=15, 
                 verbose=1,
                 validation_data=([Xtest.crop_id, Xtest.user_id], Xtest.rating))
                 
# calculate the performance on previoulsy unseen data
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(val_loss, color='b', label='Validation Loss')
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.show()

#save the model
model.save('model')

# section 2 - visualizing embedding layer

# extract embeddings
crop_em = model.get_layer('embedding')
crop_em_weights = crop_em.get_weights()[0]
crop_em_weights.shape

# get crop names form the crops.csv
crops_df_copy = crops_df.copy()
crops_df_copy = crops_df_copy.set_index("crop_id")

# write unique crop ids to tsv
c_id =list(ratings_df.crop_id.unique())
# b_id.remove(60)
dict_map = {}

for i in c_id:
    k = c_id.index(i)
    dict_map[i] = crops_df_copy.iloc[k]['name']

out_v = open('vecs.tsv', 'w')
out_m = open('meta.tsv', 'w')

for i in c_id:
    crop = dict_map[i]
    embeddings = crop_em_weights[i]
    out_m.write(crop + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    
out_v.close()
out_m.close()

# crop rating model
