import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def inference(self):
    encoder_input = keras.Input(shape=(512, 512, 1), name="one_slice")
    x = layers.Conv2D(16, 5, activation="relu", strides=2)(encoder_input)
    x = layers.LayerNormalization()(x)

    x2 = layers.Conv2D(32, 5, activation="relu", strides=2)(x)
    encoder_output = layers.LayerNormalization()(x2)
    x3 = layers.Conv2D(32, 3, activation="relu", strides=2)(encoder_output)
    encoder_output2 = layers.BatchNormalization()(x3)
    encoder = keras.Model(encoder_input, encoder_output, name="encoder")
            
    h = layers.Conv2D(32, 3, activation="relu", strides=2)(encoder_output)
    h = layers.LayerNormalization()(h)
    h = layers.Flatten()(h)
    attnn_output = layers.Dense(1)(h)
    attnder = keras.Model(encoder_input, attnn_output, name="attentionnet")
    
    use_attn = (False,True)[1]
    self.img = tf.transpose(self.img,[3,1,2,0])
    stack=tf.vectorized_map(lambda x0:encoder(tf.expand_dims(x0, axis=0)), self.img)
    if use_attn:
        attention=tf.vectorized_map(lambda x0:attnder(tf.expand_dims(x0, axis=0)), self.img)
        self.alpha=layers.Softmax()(tf.squeeze(attention,[1,2]))
        first=tf.math.reduce_sum(stack*tf.reshape(self.alpha,(-1,1,1,1,1)),axis=0)
    else:
        first=tf.math.reduce_mean(stack,axis=0)

    x = layers.Conv2D(32, 3, activation="relu", strides=2)(first)
    x = layers.BatchNormalization()(x)

    flat = layers.Flatten()(x)
    self.drop_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    dropout = layers.Dropout(self.drop_prob, name='dropout')(flat,training=True)
    self.logits = tf.squeeze(layers.Dense(2)(dropout))
    
