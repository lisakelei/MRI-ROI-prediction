import tensorflow as tf

def inference(self):
        conv0 = tf.keras.layers.Conv3D(filters=16,
                                  kernel_size=[5,5,5],
                                  padding='SAME',
                                  name='conv0')(tf.expand_dims(self.img, axis=-1))
        pool0 = tf.keras.layers.MaxPool3D(pool_size=[2, 2,1], strides=2, name='pool0')(conv0)
        n0=tf.keras.layers.BatchNormalization()(pool0)
        a0=tf.keras.layers.ReLU()(n0)
  
        conv1 = tf.keras.layers.Conv3D(filters=32,
                                  kernel_size=[5, 5,5],
                                  padding='SAME',
                                  name='conv1')(a0)
        pool1 = tf.keras.layers.MaxPool3D(pool_size=[2,2, 1], strides=2, name='pool1')(conv1)
        n1=tf.keras.layers.BatchNormalization()(pool1)
        a1=tf.keras.layers.ReLU()(n1)

        conv2 = tf.keras.layers.Conv3D(filters=32,
                                  kernel_size=[5, 5,5],
                                  strides=[2,2,1],
                                  padding='SAME',
                                  name='conv2')(a1)
        n2=tf.keras.layers.BatchNormalization()(conv2)
        a2=tf.keras.layers.ReLU()(n2)

        conv3 = tf.keras.layers.Conv3D(filters=32,
                                  kernel_size=[3,3,3],
                                  strides=2,
                                  padding='SAME',
                                  name='conv3')(a2)
        n3=tf.keras.layers.BatchNormalization()(conv3)
        a3=tf.keras.layers.ReLU()(n3)

        conv4 = tf.keras.layers.Conv3D(filters=16,
                                  kernel_size=[3, 3,3],
                                  strides=2,
                                  padding='SAME',
                                  name='conv4')(a3)
        n4=tf.keras.layers.BatchNormalization()(conv4)
        a4=tf.keras.layers.ReLU()(n4)

        conv42 = tf.keras.layers.Conv3D(filters=16,
                                  kernel_size=[3, 3,3],
                                  strides=2,
                                  padding='SAME',
                                  name='conv42')(a3)
        n42=tf.keras.layers.BatchNormalization()(conv42)
        a42=tf.keras.layers.ReLU()(n42)

        self.drop_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        dropout = tf.keras.layers.Dropout(self.drop_prob,
                                  name='dropout')(a4,training=True)
        dropout2 = tf.keras.layers.Dropout(self.drop_prob,
                                  name='dropout2')(a42,training=True)

        flat=tf.keras.layers.Flatten()(dropout)
        mean=tf.math.reduce_mean(flat,keepdims=True)
        flat2=tf.keras.layers.Flatten()(dropout2)
        mean2=tf.math.reduce_mean(flat,keepdims=True)

        self.logits=(tf.squeeze(tf.keras.layers.Dense(2)(mean))[0],tf.squeeze(tf.keras.layers.Dense(2)(mean2))[0])
