import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
# %matplotlib inline

import tensorflow as tf

# print(help(tf.lite.TFLiteConverter))

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('model')
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

