
import tensorflow as tf
from config import learning_rate, SAVE_DIR
import pickle
import os

# Placeholder for weights and biases
init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=42)

weights = {
    'wec1': tf.Variable(init(shape=[3, 1, 8])),
    'wec2': tf.Variable(init(shape=[3, 8, 16])),
    'wec3': tf.Variable(init(shape=[3, 16, 32])),
    'wef1': tf.Variable(init(shape=[4*32, 64])),
    'wef2': tf.Variable(init(shape=[64, 32])),
    'wdf1': tf.Variable(init(shape=[32, 64])),
    'wdf2': tf.Variable(init(shape=[64, 4*32])),
    'wdc1': tf.Variable(init(shape=[3, 16, 32])),
    'wdc2': tf.Variable(init(shape=[3, 8, 16])),
    'wdc3': tf.Variable(init(shape=[3, 1, 8]))
}

biases = {
    'bec1': tf.Variable(init(shape=[8])),
    'bec2': tf.Variable(init(shape=[16])),
    'bec3': tf.Variable(init(shape=[32])),
    'bef1': tf.Variable(init(shape=[64])),
    'bef2': tf.Variable(init(shape=[32])),
    'bdf1': tf.Variable(init(shape=[64])),
    'bdf2': tf.Variable(init(shape=[4*32])),
    'bdc1': tf.Variable(init(shape=[16])),
    'bdc2': tf.Variable(init(shape=[8])),
    'bdc3': tf.Variable(init(shape=[1]))
}

def save_weights_and_biases(weights, biases, fold):
    # Dynamically create file names based on fold number
    weights_file = os.path.join(SAVE_DIR, "CAE_weights_w{}.pickle".format(fold))
    biases_file = os.path.join(SAVE_DIR, "CAE_biases_w{}.pickle".format(fold))

    # Save weights
    with open(weights_file, 'wb') as handle:
        pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print   (f"Saved weights for fold {fold} at {weights_file}")

    # Save biases
    with open(biases_file, 'wb') as handle:
        pickle.dump(biases, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print  (f"Saved biases for fold {fold} at {biases_file}")

def load_weights_and_biases(fold):
    # Dynamically create file names based on fold number
    weights_file = os.path.join(SAVE_DIR, "CAE_weights_w{}.pickle".format(fold))
    biases_file = os.path.join(SAVE_DIR, "CAE_biases_w{}.pickle".format(fold))

    # Load weights if the file exists
    if os.path.exists(weights_file):
        with open(weights_file, 'rb') as handle:
            weights = pickle.load(handle)
        print  (f"Loaded weights for fold {fold} from {weights_file}")
    else:
        weights = None

    # Load biases if the file exists
    if os.path.exists(biases_file):
        with open(biases_file, 'rb') as handle:
            biases = pickle.load(handle)
        print  (f"Loaded biases for fold {fold} from {biases_file}")
    else:
        biases = None

    return weights, biases

def conv1d(x,filters_weights,bias,stride,padding):
  conv = tf.nn.conv1d(x,filters=filters_weights, stride=stride, padding=padding)
  conv = tf.add(conv, bias)
  return conv


def fully_connected(x, weight, bias):
    return tf.add(tf.matmul(x, weight), bias)




def conv1d_trans(input,filters_weights,bias,output_shape,stride,padding):
  conv_trns=tf.nn.conv1d_transpose(input,filters_weights,output_shape,stride,padding)
  conv_trns=tf.add(conv_trns,bias)
  return conv_trns



def encoder(x):
    conv1 = tf.nn.sigmoid(conv1d(x, weights['wec1'], biases['bec1'], stride= [1,2,1], padding='SAME'))
    conv2 = tf.nn.sigmoid(conv1d(conv1, weights['wec2'], biases['bec2'], stride = [1,2,1], padding='SAME'))
    conv3 = tf.nn.sigmoid(conv1d(conv2, weights['wec3'], biases['bec3'], stride= [1,2,1], padding='SAME'))

    input_fc = tf.reshape(conv3, [-1, weights['wef1'].get_shape().as_list()[0]])
    fc1 = tf.nn.sigmoid(fully_connected(input_fc, weights['wef1'], biases['bef1']))
    fc2 = tf.nn.sigmoid(fully_connected(fc1, weights['wef2'], biases['bef2']))
    return fc2


def decoder(x, num_data):
    fc1 = tf.nn.sigmoid(fully_connected(x, weights['wdf1'], biases['bdf1']))
    fc2 = tf.nn.sigmoid(fully_connected(fc1, weights['wdf2'], biases['bdf2']))

    dconv_input = tf.reshape(fc2,shape=(-1,4,32))    #replace 4 with 3 as weight initialization section explanation  (old value:(-1,4,32))
    output_shape = tf.stack([num_data, 8, 16])   #replace 8 with 5, because: ceil(18/2)=9 , ceil(9/2)=5    (old value(8,16) )
    dconv1 = tf.nn.sigmoid(conv1d_trans(dconv_input, weights['wdc1'], biases['bdc1'], output_shape, stride = [1,2,1], padding = 'SAME'))

    output_shape = tf.stack([num_data, 15, 8])  ##replace 15 with 9, because: ceil(18/2)=9       (old value:15,8)
    dconv2 = tf.nn.sigmoid(conv1d_trans(dconv1, weights['wdc2'], biases['bdc2'], output_shape, stride = [1,2,1], padding = 'SAME'))

    #output_shape = tf.stack([num_data, 18, 1])
    output_shape = tf.stack([num_data, 29, 1]) #ZS
    dconv3 = conv1d_trans(dconv2, weights['wdc3'], biases['bdc3'], output_shape, stride = [1,2,1], padding = 'SAME')

    #decode = tf.reshape(dconv3, (-1, 18))
    decode = tf.reshape(dconv3, (-1, 29))

    return decode



def loss(x,y):
    x=tf.reshape(x,(-1,29))
    #x=tf.reshape(x,(-1,18))

    return tf.reduce_mean(tf.square(x-y))


def gradiant(x):
    with tf.GradientTape() as tape:
        decode = decoder(encoder(x), x.shape[0])
        loss_ = loss(x, decode)
        trainable_variables = list(weights.values()) + list(biases.values())
        gradiants = tape.gradient(loss_, trainable_variables)

        optimzer = tf.optimizers.Adam(learning_rate)
        optimzer.apply_gradients(zip(gradiants, trainable_variables))

    # Define classification metrics
