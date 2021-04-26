import tensorflow as tf
import numpy as np
import os
from math import sqrt
from utils import normalize

def append_input(filename,NUMTRAIN_SAMPLES,axis=1,dtype=np.uint8):
    
    y=[]
    for i, f in enumerate(filename):
     x=np.fromfile(f,dtype=dtype)
     x=np.reshape(x, [NUMTRAIN_SAMPLES,-1])
     if i==0:
         y=x
     else:       
         y=np.concatenate((y,x), axis = axis)
    PATH=os.makedir('./data/')
    y.tofile([PATH+'stack_train_data.bin'])


#Input pipeline train
def load(value,record_bytes,first_dim):

  record= tf.io.decode_raw(value, tf.uint8)
  depth_label_major = (tf.reshape(record,
      [1,int(first_dim), int(record_bytes/first_dim)]))
  image=tf.transpose(depth_label_major,[1,2,0])
  image=normalize(tf.cast(image,tf.float32))
  return image


#Input pipeline test
# def load_test(value,record_bytes):
    
#   record= tf.io.decode_raw(value, tf.uint8)
#   depth_label_major = (tf.reshape(record,
#       [1,int(sqrt(record_bytes)), int(sqrt(record_bytes))]))
#   image=tf.transpose(depth_label_major,[1,2,0])
#   image=normalize(tf.cast(image,tf.float32))
#   return image