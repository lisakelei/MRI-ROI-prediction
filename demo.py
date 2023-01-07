import os
import numpy as np
import time
import glob
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('EXP','newattn2AsS/ckpt-80546',"exp and ckpt name")
tf.compat.v1.flags.DEFINE_integer('mod', 0, "model") # 0=share, 1=chstack, 2=3D


class ConvNet(object):
    def __init__(self):
        self.lr = 0.0001
        self.batch_size = 1
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.image_size=512
   

    if FLAGS.mod==1:
        from bmbn2D import inference
    elif FLAGS.mod==2:
        from bmbn import inference
    else:
        from share import inference


    def parser(self,serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.io.parse_single_example(serialized_example,
                        features={
                            'top': tf.io.FixedLenFeature([], tf.float32),
                            'bottom': tf.io.FixedLenFeature([], tf.float32),
                            'right': tf.io.FixedLenFeature([], tf.float32),
                            'left': tf.io.FixedLenFeature([], tf.float32),
                            'image': tf.io.FixedLenFeature([], tf.string),
                        }, name='features')
        image = tf.io.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [self.image_size,self.image_size,-1])
        label = tf.stack([features['top'],features['bottom'],features['right'],features['left']])
        return image,label


    def get_data(self):
        with tf.name_scope('data'):
            self.filenames = tf.compat.v1.placeholder(tf.string, shape=[None])
            dataset = tf.data.TFRecordDataset(self.filenames)
            dataset=dataset.map(self.parser,num_parallel_calls=4)
            if FLAGS.mod!=1:
                dataset=dataset.batch(1)
            else:
                dataset=dataset.padded_batch(self.batch_size,padded_shapes=([512,512,40],[4]))
            self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            self.img, self.label= self.iterator.get_next()
            self.img=tf.image.per_image_standardization(self.img)

    def build(self):
        self.get_data()
        self.inference()

    def eval_once(self, sess, init):
        eval_filenames=sorted(glob.glob("./test/*.tfrecord"))
        start_time = time.time()
        sess.run(init.initializer, feed_dict={self.filenames:eval_filenames}) 
        scores=[]
        truepf=[]
        IoUs=[]
        alps=[]
        try:
            while True:
                score,btrue_pf= sess.run([self.logits,self.label], feed_dict={self.drop_prob:0.0})
                score=[max(0,score[0]),min(512.0, score[1])]
                scores+=[score[0],score[1]]
                #truepf+=[btrue_pf[0][1],btrue_pf[0][0]]
                truepf+=[btrue_pf[0][3],btrue_pf[0][2]]
                #IoUs+=[(min(score[1],btrue_pf[0][0])-max(score[0],btrue_pf[0][1]))/(max(score[1],btrue_pf[0][0])-min(score[0],btrue_pf[0][1]))]
                IoUs+=[(min(score[1],btrue_pf[0][2])-max(score[0],btrue_pf[0][3]))/(max(score[1],btrue_pf[0][2])-min(score[0],btrue_pf[0][3]))]
        except tf.errors.OutOfRangeError:
            pass
        end_time = time.time()
        print('TIME ', end_time-start_time)
        print(eval_filenames)
        print('score= ', scores, 'label= ', truepf)
        pf_error=np.mean(abs(np.array(scores)-np.array(truepf)))
        IoU=np.mean(np.array(IoUs))
        print('IoU= ', IoUs)
        return pf_error,IoU

    def train(self):
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, 'checkpoints/'+FLAGS.EXP)
            pf_error,iou=self.eval_once(sess, self.iterator)
            print('DONE with error ', pf_error, iou)

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train()

