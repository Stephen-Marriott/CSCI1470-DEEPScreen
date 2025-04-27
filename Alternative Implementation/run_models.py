#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 17:12:51 2025

@author: wwelsh
"""


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import json
from original_model_tf import DeepScreen_Tensorflow
from alt_CNN_tf import Alternative_CNN
from attention_CNN import Attention_CNN


def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img= tf.cast(img, tf.float32)
    img = tf.math.divide(img, 255.)

    return img

def load_and_concatenate_images(folder_path,dictionary):
    image_tensors = []
    labels = []
    training_proteins = list(dictionary.keys())
    for t in training_proteins:
        file_path =  folder_path + t +".png"
        if os.path.isfile(file_path):
            image_tensor = load_image(file_path)
        if image_tensor is not None:
            image_tensors.append(image_tensor)
            labels.append(dictionary.get(t))
    
    image_tensors = tf.convert_to_tensor(image_tensors)
    labels = tf.convert_to_tensor(labels)
    
    return image_tensors,labels
    

def get_protein_data(protein_name,data_dir):
    label_file_path = data_dir+protein_name+r"/train_val_test_dict.json"
    with open(label_file_path, 'r') as f:
        data = json.load(f)
        
    img_path = data_dir+protein_name+r"/imgs/"

    training_data = dict(data.get('training'))
    validation_data = dict(data.get('validation'))
    testing_data = dict(data.get('test'))
    
    train_imgs,train_labels = load_and_concatenate_images(img_path, training_data)
    val_imgs,val_labels = load_and_concatenate_images(img_path, validation_data)
    test_imgs,test_labels = load_and_concatenate_images(img_path, testing_data)
    
    train_labels = tf.one_hot(train_labels, 2) 
    val_labels = tf.one_hot(val_labels, 2) 
    test_labels = tf.one_hot(test_labels, 2) 
    
    return train_imgs,train_labels,val_imgs,val_labels,test_imgs,test_labels


def creat_and_run_model(model_type,
                protein_name,
                model_dir,
                training_images,
                training_labels,
                validation_images,
                validation_labels,
                test_images,
                test_labels,
                fc_size_1 = 256,
                fc_size_2 = 128,
                dropout_rate = .25,
                epochs = 20,
                batch_size = 32,
                optimizer_choice = 'adam',
                vb = 0):
    
    
    model_fp = model_dir + protein_name +'-' +model_type+'.keras'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_fp,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 verbose=0)

    callbacks_list = [checkpoint]

    if model_type == 'DEEPScreen':
        model = DeepScreen_Tensorflow(fc_size_1,fc_size_2,dropout_rate)
        
    elif model_type == 'Alternative_CNN':
        model = Alternative_CNN(fc_size_1,fc_size_2,dropout_rate)
    elif model_type == 'Attention_CNN':
        model = Attention_CNN(fc_size_1,fc_size_2,dropout_rate)
        model.build(input_shape=(None, 200, 200, 3))
    else:
        raise ValueError("No Specified Model Type")
        
    model.compile(optimizer=optimizer_choice, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics = ['accuracy','precision'])


    history = model.fit(
        x=training_images,
        y=training_labels,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data = [validation_images,validation_labels],
        callbacks=callbacks_list,
        verbose=vb
        )

    best_model = tf.keras.models.load_model(model_fp)

    loss, accuracy, precision = best_model.evaluate(test_images, test_labels, verbose=0)

    print(f'Test Loss of Best Model: {loss:.4f}')
    print(f'Test Accuracy of Best Model: {accuracy:.4f}')
    print(f'Test Precision of Best Model: {precision:.4f}')

    
    return loss, accuracy, precision


def process_target_protein(protein_name,
                           data_dir,
                           model_dir,
                           model_type,
                           fc_size_1 = 256,
                           fc_size_2 = 128,
                           dropout_rate = .25,
                           epochs = 20,
                           batch_size = 32,
                           optimizer_choice = 'adam',
                           vb = 0):
    
    
    train_imgs,train_labels,val_imgs,val_labels,test_imgs,test_labels = get_protein_data(protein_name, data_dir)
    
    loss,accuracy,precision = creat_and_run_model(model_type,
                                                  protein_name,
                                                  model_dir,
                                                  train_imgs,
                                                  train_labels,
                                                  val_imgs,
                                                  val_labels,
                                                  test_imgs,
                                                  test_labels,
                                                  fc_size_1,
                                                  fc_size_2,
                                                  dropout_rate,
                                                  epochs,
                                                  batch_size,
                                                  optimizer_choice,
                                                  vb)
    
    result_dict = {'Target Protein':protein_name,
                   'Model Type': model_type,
                   'Test Loss':loss,
                   'Test Accuracy':accuracy,
                   'Test Precision':precision}
    
    return result_dict
    
    

protein_list = ["CHEMBL1862"]
    
dd = r"/Users/wwelsh/Downloads/"
mp = r"/Users/wwelsh/Documents/GitHub/working_final_project_1470/Models/"
model_types = ['DEEPScreen','Alternative_CNN','Attention_CNN']
#model_types = ['Attention_CNN']
result_list = []

for p in protein_list:
    print(f'Modeling Protein: {p}')
    for m in model_types:
        print(f'Model Type: {m}')
        result = process_target_protein(p, dd, mp, m,vb=1,epochs=10)
        result_list.append(result)

result_df = pd.DataFrame(result_list)

    
    
    