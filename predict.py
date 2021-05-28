import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow.keras as tf

# load ratings dataframe
ratings_df = pd.read_csv("crop-data/ratings.csv")
# load crops dataframe
crops_df = pd.read_csv("crop-data/crops.csv")

# load trained model
model = tf.models.load_model('model')

# crop ids list
c_id =list(ratings_df.crop_id.unique())
# c_id.remove(60)

user_id = 314

# making recommendations for user 315
crop_arr = np.array(c_id) #get all crop IDs
user = np.array([user_id for i in range(len(c_id))])
pred = model.predict([crop_arr, user])
# print(pred)

# retrieve corresponding crops from the dataset
pred = pred.reshape(-1) #reshape to single dimension
pred_ids = (-pred).argsort()[0:5]
# print(pred_ids)

# use the index to retrive crops dataframe
pred_df = crops_df.iloc[pred_ids]

# convert to JSON format for web
web_crop_data = pred_df[["crop_id", "name", "crop_category", "timestamp"]]
web_crop_data = web_crop_data.sort_values('crop_id')
web_crop_data.head()

print(f"Recommended crops for user {user_id} are: ")
print(web_crop_data)

# export JSON
web_crop_data.to_json(r'crop_data.json', orient='records')

# crop recommendations
