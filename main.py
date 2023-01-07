import os
import numpy as np
import time
import glob
import random
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('EXP','temp',"exp. name")
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
            dataset=dataset.shuffle(100)
            self.iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            self.img, self.label= self.iterator.get_next()
            self.img=tf.image.per_image_standardization(self.img)
            self.shift = tf.compat.v1.placeholder(tf.int32, name='shift')
            self.img=tf.roll(self.img,self.shift,[0,1])
            self.label+=tf.cast(self.shift[0],tf.float32)

    def loss(self):
        with tf.name_scope('loss'):
            '''toploss=tf.where(self.label[0,0]>self.logits[1], 
                    2*tf.keras.losses.MSE(self.label[:,0],self.logits[1]), 
                    tf.keras.losses.MSE(self.label[:,0],self.logits[1]))
            bottomloss=tf.where(self.label[0,1]<self.logits[0],
                    2*tf.keras.losses.MSE(self.label[:,1],self.logits[0]), 
                    tf.keras.losses.MSE(self.label[:,1],self.logits[0]))
            self.loss=toploss+bottomloss'''
            self.loss=tf.keras.losses.MSE(self.label[:,1],(self.logits[0]))+tf.keras.losses.MSE(self.label[:,0],self.logits[1])  
    
    def optimize(self):
        self.opt = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss, 
                global_step=self.gstep)

    def summary(self):
        with tf.name_scope('summaries'):
            tf.compat.v1.summary.scalar('loss', self.loss)
            self.summary_op = tf.compat.v1.summary.merge_all()

    def build(self):
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        train_filenames=sorted(glob.glob("/mnt/raid5/kllei/Loc/trainab/*.tfrecord"))
        sess.run(init.initializer, feed_dict={self.filenames: train_filenames}) 
        try:
            while True:
                shiftof=[-30,-20,-10,0,10,20,30]
                shiftofx=[-6,-3,0,3,6]
                #feedalp=np.(self.img.shape)
                dx=random.choice(shiftofx)
                dy=random.choice(shiftof)
                _, l, summaries,tsnr,tscore,img = sess.run([self.opt, self.loss, self.summary_op,self.label,self.logits,self.img], feed_dict={self.drop_prob:0.2, self.shift:[dy,dx]})#self.alpha:feedalp
                writer.add_summary(summaries, global_step=step)
                if step % 100 == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
        except tf.errors.OutOfRangeError:
            pass
        return step

    def eval_once(self, sess, init, writer, step):
        eval_filenames=sorted(glob.glob("./testab/*.tfrecord"))
        sess.run(init.initializer, feed_dict={self.filenames:eval_filenames}) 
        scores=[]
        truepf=[]
        IoUs=[]
        hIoUs=[]
        try:
            while True:
                score,btrue_pf= sess.run([self.logits,self.label], feed_dict={self.drop_prob:0.0, self.shift:[0,0]})
                scores+=[score[0],score[1]]
                truepf+=[btrue_pf[0][1],btrue_pf[0][0]]
                IoUs+=[(min(score[1],btrue_pf[0][0])-max(score[0],btrue_pf[0][1]))/(max(score[1],btrue_pf[0][0])-min(score[0],btrue_pf[0][1]))]
        except tf.errors.OutOfRangeError:
            pass
        print('score= ', scores, 'label= ', truepf)
        pf_error=np.mean(abs(np.array(scores)-np.array(truepf)))
        IoU=np.mean(np.array(IoUs))
        evalsum = tf.compat.v1.Summary()
        evalsum.value.add(tag='pf_error', simple_value=pf_error)
        evalsum.value.add(tag='IoU', simple_value=IoU)
        writer.add_summary(evalsum, global_step=step)
        return pf_error

    def train(self, n_epochs):
        try:
            os.mkdir('checkpoints/'+FLAGS.EXP)
        except:
            pass
        writer = tf.compat.v1.summary.FileWriter('./graphs/'+FLAGS.EXP, tf.compat.v1.get_default_graph())
        config = tf.compat.v1.ConfigProto(log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state('checkpoints/'+FLAGS.EXP)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            
            step = self.gstep.eval()
            best_error=200
            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.iterator, writer, epoch, step)
                if (epoch+1) % 10 == 1:
                    pf_error=self.eval_once(sess, self.iterator, writer, step)
                    if pf_error<=best_error:
                        best_error=min(pf_error,best_error)
                        saver.save(sess, 'checkpoints/'+FLAGS.EXP+'/ckpt', step)
            saver.save(sess, 'checkpoints/'+FLAGS.EXP+'/ckpt', step)
            print('DONE with best error ',best_error)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=3000)
