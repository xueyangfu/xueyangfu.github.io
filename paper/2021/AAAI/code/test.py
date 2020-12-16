#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This is a implementation of testing code of this paper:
# X. Fu, Q. Qi, Y. Zhu, X. Ding and Z.-J. Zha. “Rain Streak Removal via Dual Graph Convolutional Network”, AAAI, 2021.
# author: Xueyang Fu (xyfu@ustc.edu.cn)


import os
import skimage.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import model

tf.reset_default_graph()



model_path = './TrainedModel/model-Rain200H' # trained model

input_path = './TestImg/rain/' # the path of testing images

results_path = './TestImg/results/'  # the path of de-rained results



def _parse_function(filename):      
  image_string = tf.read_file(filename)  
  image_decoded = tf.image.decode_png(image_string, channels = 3)  
  rain = tf.cast(image_decoded, tf.float32)/255.
  return rain



if __name__ == '__main__':
  
   imgName = os.listdir(input_path)      
   filename = os.listdir(input_path)
   for i in range(len(filename)):
      filename[i] = input_path + filename[i]
      
    
   filename_tensor = tf.convert_to_tensor(filename, dtype = tf.string)        
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.prefetch(buffer_size = 10)
   dataset = dataset.batch(1).repeat()  
   iterator = dataset.make_one_shot_iterator()  
   rain = iterator.get_next()    

   imglist = model.Inference(rain)
   final = tf.clip_by_value(imglist, 0., 1.)
   final = final[0,:,:,:]


   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True
   
   
   num_img = len(filename)
   with tf.Session(config = config) as sess:

        all_vars = tf.trainable_variables() 
        all_vars = tf.train.Saver(var_list = all_vars)
        all_vars.restore(sess,model_path)          
        print ("Loading model")

        for i in range(num_img):
     
            derained, ori = sess.run([final, rain])              
            derained = np.uint8(derained* 255.)
     
            index = imgName[i].rfind('.')
            name = imgName[i][:index]
            skimage.io.imsave(results_path + name +'.png', derained) 
            print('Processing %d / %d image' % (i+1,num_img))    
            
   sess.close()   
   
   plt.subplot(1,2,1)     
   plt.imshow(ori[0,:,:,:])          
   plt.title('Rainy input')
   plt.subplot(1,2,2)    
   plt.imshow(derained)
   plt.title('De-rained result')
   plt.show()        