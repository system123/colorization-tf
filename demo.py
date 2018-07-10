import tensorflow as tf
from utils import *
from net import Net
from skimage.io import imsave
from skimage.transform import resize
import numpy as np
import cv2
import os

out_dir = './'
filename = 'datum/lab/ROIs1970_fall_129_p774.png'

img = cv2.imread(filename)
if len(img.shape) == 3:
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = img[None, :, :, None]
data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

#data_l = tf.placeholder(tf.float32, shape=(None, None, None, 1))
autocolor = Net(train=False, ret_layer='conv7_3')

conv7_3 = autocolor.inference(data_l)

saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, 'models/model.ckpt')
  conv7_3 = sess.run(conv7_3)

npz_fn = os.path.basename(filename).replace('png', 'npz')
npz_fn = os.path.join(out_dir, npz_fn)
np.savez_compressed(npz_fn, conv7_3)
# img_rgb = decode(data_l, conv7_3, 2.63)
# imsave('color.jpg', img_rgb)
