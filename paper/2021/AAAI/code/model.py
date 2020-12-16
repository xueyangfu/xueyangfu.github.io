#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This is a implementation of testing code of this paper:
# X. Fu, Q. Qi, Y. Zhu, X. Ding and Z.-J. Zha. “Rain Streak Removal via Dual Graph Convolutional Network”, AAAI, 2021.
# author: Xueyang Fu (xyfu@ustc.edu.cn)

import numpy as np
import tensorflow as tf



def spatialGCN(input_tensor):  
    
    in_channels = input_tensor.get_shape().as_list()[-1]    
    inputs_shape = tf.shape(input_tensor)
    
    channels = in_channels//2

    theta = tf.layers.conv2d(input_tensor, channels, 1, padding="valid")
    theta = tf.reshape(theta, shape=[-1, inputs_shape[1] * inputs_shape[2], channels])

    nu =  tf.layers.conv2d(input_tensor, channels, 1, padding="valid")
    nu =  tf.reshape(nu, shape=[-1, tf.shape(input_tensor)[1] * tf.shape(input_tensor)[2], channels])
    nu_tmp = tf.reshape(nu, shape=[-1,  tf.shape(nu)[1] *  tf.shape(nu)[2]])    
    nu_tmp = tf.nn.softmax(nu_tmp, axis=-1)
    nu = tf.reshape(nu_tmp, shape=[-1, tf.shape(nu)[1],  tf.shape(nu)[2]])  
    
    xi =  tf.layers.conv2d(input_tensor, channels, 1, padding="valid")
    xi =  tf.reshape(xi, shape=[-1, inputs_shape[1] * inputs_shape[2], channels])
    xi_tmp = tf.reshape(xi, shape=[-1,  tf.shape(xi)[1] *  tf.shape(xi)[2]])    
    xi_tmp = tf.nn.softmax(xi_tmp, axis=-1)
    xi = tf.reshape(xi_tmp, shape=[-1, tf.shape(xi)[1],  tf.shape(xi)[2]])     


    F_s = tf.matmul(nu, xi, transpose_a=True)                
    AF_s = tf.matmul(theta, F_s)  
    
    AF_s = tf.reshape(AF_s, shape=[-1,  inputs_shape[1],  inputs_shape[2], channels])

    F_sGCN = tf.layers.conv2d(AF_s, in_channels, 1, padding="valid")

    return F_sGCN + input_tensor


def channelGCN(input_tensor):  
    
    input_chancel = input_tensor.get_shape().as_list()[-1]
    inputs_shape = tf.shape(input_tensor)
    
    C =  input_chancel//2
    N =  input_chancel//4
    
    zeta = tf.layers.conv2d(input_tensor, C, 1, padding="valid")
    zeta = tf.reshape(zeta, [inputs_shape[0], -1, C]) 

    kappa = tf.layers.conv2d(input_tensor, N, 1, padding="valid")
    kappa = tf.reshape(kappa, [inputs_shape[0], -1, N]) 
    kappa = tf.transpose(kappa, perm=[0, 2, 1])  


    F_c = tf.matmul(kappa, zeta)
    
    F_c_tmp = tf.reshape(F_c, shape=[-1,  C * N])    
    F_c_tmp = tf.nn.softmax(F_c_tmp, axis=-1)
    F_c = tf.reshape(F_c_tmp, shape=[-1, N,  C]) 
    
    F_c = tf.expand_dims(F_c, axis=1)  
    F_c = F_c + tf.layers.conv2d(F_c, C, 1, padding="valid")          
    F_c = tf.nn.relu(F_c) 
    F_c = tf.transpose(F_c, perm=[0, 3, 1, 2]) 

        
    F_c = tf.layers.conv2d(F_c, N, 1, padding="valid")    
    F_c = tf.reshape(F_c, [inputs_shape[0], C, N])  
    F_c = tf.matmul(zeta, F_c)
    
    
    F_c = tf.expand_dims(F_c, axis=1)  
    F_c = tf.reshape(F_c, [inputs_shape[0], inputs_shape[1], inputs_shape[2], N]) 
    F_cGCN = tf.layers.conv2d(F_c, input_chancel, 1, padding="valid")
    
    return  F_cGCN + input_tensor



def BasicUnit(input_tensor): 
    
    channels = input_tensor.get_shape().as_list()[-1]  
  
    F_sGCN = spatialGCN(input_tensor)
    
    conv1 = tf.layers.conv2d(F_sGCN, channels, 3, dilation_rate=(1, 1), padding="SAME",activation = tf.nn.relu)     
    conv2 = tf.layers.conv2d(conv1, channels, 3, dilation_rate=(1, 1), padding="SAME",activation = tf.nn.relu)   
    conv3 = tf.layers.conv2d(F_sGCN, channels, 3, dilation_rate=(3, 3), padding="SAME",activation = tf.nn.relu)   
    conv4 = tf.layers.conv2d(conv3, channels, 3, dilation_rate=(3, 3), padding="SAME",activation = tf.nn.relu)      
    tmp = tf.concat([F_sGCN, conv1, conv2, conv3, conv4],-1)    
    
    F_DCM = tf.layers.conv2d(tmp, channels, 1, padding="SAME",activation = tf.nn.relu)     
    
    F_cGCN = channelGCN(F_DCM)
    
    F_unit = F_cGCN + input_tensor
    
    return  F_unit



def Inference(images, channels = 72):
    
  inchannels = images.get_shape().as_list()[-1]  
  
  with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):       
         
     with tf.variable_scope('basic'):                     
          basic_fea0 = tf.layers.conv2d(images, channels, 3, padding="SAME")       
          basic_fea1 = tf.layers.conv2d(basic_fea0, channels, 3, padding="SAME")             
                 
     with tf.variable_scope('encoder0'):         
          encode0 =  BasicUnit(basic_fea1) 
         
     with tf.variable_scope('encoder1'):              
          encode1 =  BasicUnit(encode0) 
        

     with tf.variable_scope('encoder2'): 
          encode2 =  BasicUnit(encode1) 
         
     with tf.variable_scope('encoder3'): 
          encode3 =  BasicUnit(encode2)        


     with tf.variable_scope('encoder4'): 
          encode4 =  BasicUnit(encode3)      


     with tf.variable_scope('middle'):         
          middle_layer =   BasicUnit(encode4)


     with tf.variable_scope('decoder4'):              
          decoder4 = tf.concat([middle_layer, encode4] ,-1)
          decoder4 = tf.layers.conv2d(decoder4, channels, 1, padding="SAME")  
          decoder4 = BasicUnit(decoder4)
          
          
     with tf.variable_scope('decoder3'):              
          decoder3 = tf.concat([decoder4, encode3] ,-1)
          decoder3 = tf.layers.conv2d(decoder3, channels, 1, padding="SAME")  
          decoder3 = BasicUnit(decoder3)

                                  
     with tf.variable_scope('decoder2'):              
          decoder2 = tf.concat([decoder3, encode2] ,-1)
          decoder2 = tf.layers.conv2d(decoder2, channels, 1, padding="SAME")  
          decoder2 = BasicUnit(decoder2)
         
         
     with tf.variable_scope('decoder1'):              
          decoder1 = tf.concat([decoder2, encode1] ,-1)         
          decoder1 = tf.layers.conv2d(decoder1, channels, 1, padding="SAME")  
          decoder1 = BasicUnit(decoder1)
         
         
     with tf.variable_scope('decoder0'):              
          decoder0 = tf.concat([decoder1, encode0] ,-1) 
          decoder0 = tf.layers.conv2d(decoder0, channels, 1, padding="SAME")
          decoder0 = BasicUnit(decoder0) 
            

     with tf.variable_scope('reconstruct'):      
          decoding_end = tf.concat([decoder0, basic_fea1] ,-1) 
          decoding_end = tf.layers.conv2d(decoding_end, channels, 3, padding="SAME", activation = tf.nn.relu)                
                          
          decoding_end = decoding_end + basic_fea0
          res =  tf.layers.conv2d(decoding_end, inchannels, 3, padding = 'SAME') 
          output  = images + res    

  return  output




 
if __name__ == '__main__':
    tf.reset_default_graph()
    input_x = tf.placeholder(tf.float32, [1,101,101,3])
    output = Inference(input_x)      
    all_vars = tf.trainable_variables()
    print("Total parameters' number: %d" 
         %(np.sum([np.prod(v.get_shape().as_list()) for v in all_vars])))  