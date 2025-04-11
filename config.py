import os
import numpy as np
import tensorflow as tf
import random
import pickle

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

batch_size = 128
learning_rate = 0.001
train_step = 150
SAVE_DIR = "./saved_weights_biases"
os.makedirs(SAVE_DIR, exist_ok=True)
