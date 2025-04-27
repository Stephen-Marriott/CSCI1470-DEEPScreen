#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:57:42 2025

@author: wwelsh
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import json

#Taken from https://github.com/laugh12321/3D-Attention-Keras
class spatial_attention(tf.keras.layers.Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    def __init__(self, kernel_size=2, **kwargs):
        self.kernel_size = kernel_size
        super(spatial_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=self.kernel_size,
                                             strides=1, padding='same', activation='sigmoid',
                                             kernel_initializer='he_normal', use_bias=False)
        super(spatial_attention, self).build(input_shape)


    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        

        concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
        feature = self.conv2d(concat)	
            
        return tf.keras.layers.multiply([inputs, feature])


@tf.keras.utils.register_keras_serializable()
class Attention_CNN(tf.keras.Model):

    def __init__(self, fully_layer_1, fully_layer_2, drop_rate):
        super(Attention_CNN, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(32, 2)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.act1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(64, 2)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.act2 = tf.keras.layers.ReLU()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        self.conv3 = tf.keras.layers.Conv2D(64, 2)
        self.bn3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.act3 = tf.keras.layers.ReLU()
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        self.spatial_attention = spatial_attention()
        self.drop_rate = drop_rate
        
        self.fl = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(fully_layer_1, activation='relu')
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        self.fc2 = tf.keras.layers.Dense(fully_layer_2, activation='relu')
        self.drop2 = tf.keras.layers.Dropout(drop_rate)
        self.fc3 = tf.keras.layers.Dense(2)
        
        self.drop3 = tf.keras.layers.Dropout(drop_rate)
        self.drop4 = tf.keras.layers.Dropout(drop_rate)
        
    def call(self, inputs):
        

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop3(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.act2(x)
        # x = self.pool2(x)

        # feature_map = self.conv3(x)
        # attention_map = self.attention_map(x)
        # x = self.mult1([feature_map, attention_map])
        x = self.spatial_attention(x)
        x = self.fl(x)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        outputs = self.fc3(x)

        
        return outputs

    
    def get_config(self):
        config = super().get_config()
        config.update({
            "fully_layer_1": self.fc1.units,
            "fully_layer_2": self.fc2.units,  
            "drop_rate": self.drop_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(config["fully_layer_1"], config["fully_layer_2"], config["drop_rate"])