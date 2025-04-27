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


@tf.keras.utils.register_keras_serializable()
class Alternative_CNN(tf.keras.Model):

    def __init__(self, fully_layer_1, fully_layer_2, drop_rate):
        super(Alternative_CNN, self).__init__()
        
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

        # self.conv4 = tf.keras.layers.Conv2D(128, 2)
        # self.bn4 = tf.keras.layers.BatchNormalization(axis=-1)
        # self.act4 = tf.keras.layers.ReLU()
        # self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        # self.conv5 = tf.keras.layers.Conv2D(32, 2)
        # self.bn5 = tf.keras.layers.BatchNormalization(axis=-1)
        # self.act5 = tf.keras.layers.ReLU()
        # self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

      

        self.drop_rate = drop_rate
        
        self.fl = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(fully_layer_1, activation='relu')
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        self.fc2 = tf.keras.layers.Dense(fully_layer_2, activation='relu')
        self.drop2 = tf.keras.layers.Dropout(drop_rate)
        self.fc3 = tf.keras.layers.Dense(2)

        self.model = tf.keras.Sequential([
            self.conv1,
            self.bn1,
            self.act1,
            self.pool1,
            self.conv2,
            self.bn2,
            self.act2,
            self.pool2,
            self.conv3,
            self.bn3,
            self.act3,
            self.pool3,
            # self.conv4,
            # self.bn4,
            # self.act4,
            # self.pool4,
            # self.conv5,
            # self.bn5,
            # self.act5,
            # self.pool5,
            self.fl,
            self.fc1,
            self.drop1,
            self.fc2,
            self.drop2,
            self.fc3
            
        ])
        
        self.model.build(input_shape=(None, 200, 200, 3))


        
    def call(self, inputs):
        
        return self.model(inputs)

    
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