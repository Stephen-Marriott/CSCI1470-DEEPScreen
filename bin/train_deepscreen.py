import os
import sys
import cv2
import json
import random
import warnings
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from models import CNNModel1
from data_processing import get_train_test_val_data_loaders
from evaluation_metrics import prec_rec_f1_acc_mcc, get_list_of_scores

warnings.filterwarnings(action='ignore')
np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

project_file_path = "{}DEEPScreen".format(os.getcwd().split("DEEPScreen")[0])
training_files_path = "{}/training_files".format(project_file_path)
result_files_path = "{}/result_files".format(project_file_path)
trained_models_path = "{}/trained_models".format(project_file_path)

def save_best_model_predictions(experiment_name, epoch, validation_scores_dict, test_scores_dict, model, project_file_path, target_id, str_arguments,
                                                                                   all_test_comp_ids, test_labels, test_predictions):

    # Ensure the directory exists
    experiment_dir = os.path.join(trained_models_path, experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Save the TensorFlow model
    model_save_path = os.path.join(experiment_dir, f"{target_id}_{str_arguments}_best_val_epoch_{epoch}")
    model.save(model_save_path)

    # print(all_test_comp_ids)
    str_test_predictions = "CompoundID\tLabel\tPred\n"
    for ind in range(len(all_test_comp_ids)):
        str_test_predictions += "{}\t{}\t{}\n".format(all_test_comp_ids[ind],
                                                          test_labels[ind],
                                                          test_predictions[ind])

    best_test_performance_dict = test_scores_dict
    best_test_predictions = str_test_predictions

    return validation_scores_dict, best_test_performance_dict, best_test_predictions, str_test_predictions

def get_device():
    device = "CPU"
    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        print("GPU is available on this device!")
        device = "GPU"
    else:
        print("CPU is available on this device!")

    return device

def calculate_val_test_loss(model, criterion, data_loader, device):
    total_count = 0
    total_loss = 0.0
    all_comp_ids = []
    all_labels = []
    all_predictions = []

    for i, data in enumerate(data_loader):
        img_arrs, labels, comp_ids = data
        img_arrs, labels = tf.convert_to_tensor(img_arrs, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int32)

        total_count += len(comp_ids)

        y_pred = model(img_arrs)

        loss = criterion(labels, y_pred)
        total_loss += float(loss.numpy())

        all_comp_ids.extend(list(comp_ids))

        preds = tf.argmax(y_pred, axis=1).numpy()

        all_labels.extend(list(labels.numpy()))
        all_predictions.extend(list(preds))


    return total_loss, total_count, all_comp_ids, all_labels, all_predictions


def train_validation_test_training(target_id, model_name, fully_layer_1, fully_layer_2, learning_rate, batch_size, drop_rate, n_epoch, experiment_name):
    arguments = [str(argm) for argm in
                 [target_id, model_name, fully_layer_1, fully_layer_2, learning_rate, batch_size, drop_rate, n_epoch, experiment_name]]

    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    device = get_device()
    exp_path = os.path.join(result_files_path, "experiments", experiment_name)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    best_val_test_result_fl = open(
        "{}/best_val_test_performance_results-{}.txt".format(exp_path,str_arguments), "w")
    best_val_test_prediction_fl = open(
        "{}/best_val_test_predictions-{}.txt".format(exp_path,str_arguments), "w")

    train_loader, valid_loader, test_loader = get_train_test_val_data_loaders(target_id, batch_size)
    model = None
    if model_name == "CNNModel1":
        model = CNNModel1(fully_layer_1, fully_layer_2, drop_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    best_val_mcc_score, best_test_mcc_score = 0.0, 0.0
    best_val_test_performance_dict = dict()
    best_val_test_performance_dict["MCC"] = 0.0
    best_test_performance_dict = dict()
    best_test_performance_dict["MCC"] = 0.0

    for epoch in range(n_epoch):
        total_training_count = 0
        total_training_loss = 0.0
        print("Epoch :{}".format(epoch))
        batch_number = 0
        all_training_labels = []
        all_training_preds = []

        for i, data in enumerate(train_loader):
            batch_number += 1
            # print(batch_number)
            # clear gradient DO NOT forget you fool!
            img_arrs, labels, comp_ids = data
            img_arrs, labels = tf.convert_to_tensor(img_arrs, dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.int32)

            total_training_count += len(comp_ids)

            with tf.GradientTape() as tape:
                y_pred = model(img_arrs, training=True)
                loss = criterion(labels, y_pred)

            total_training_loss += loss.numpy()

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            preds = tf.argmax(y_pred, axis=1).numpy()
            all_training_labels.extend(list(labels.numpy()))
            all_training_preds.extend(list(preds))

        print(f"Labels: {labels}")
        print(f"Prediction: {preds}")

        print("Epoch {} training loss:".format(epoch), total_training_loss)

        training_perf_dict = dict()
        try:
        	training_perf_dict = prec_rec_f1_acc_mcc(all_training_labels, all_training_preds)
        except:
        	print("There was a problem during training performance calculation!")
        # print(training_perf_dict)

        total_val_loss, total_val_count, all_val_comp_ids, all_val_labels, val_predictions = calculate_val_test_loss(model, criterion, valid_loader, device)
        val_perf_dict = dict()
        val_perf_dict["MCC"] = 0.0
        try:
            val_perf_dict = prec_rec_f1_acc_mcc(all_val_labels, val_predictions)
        except:
            print("There was a problem during validation performance calculation!")


        total_test_loss, total_test_count, all_test_comp_ids, all_test_labels, test_predictions = calculate_val_test_loss(
            model, criterion, test_loader, device)
        test_perf_dict = dict()
        test_perf_dict["MCC"] = 0.0
        try:
            test_perf_dict = prec_rec_f1_acc_mcc(all_test_labels, test_predictions)
        except:
            print("There was a problem during test performance calculation!")

        if epoch == 0 or val_perf_dict["MCC"] > best_val_mcc_score:
            best_val_mcc_score = val_perf_dict["MCC"]
            best_test_mcc_score = test_perf_dict["MCC"]

            validation_scores_dict, best_test_performance_dict, best_test_predictions, str_test_predictions = save_best_model_predictions(
                experiment_name, epoch, val_perf_dict, test_perf_dict,
                model,project_file_path, target_id, str_arguments,
                all_test_comp_ids, all_test_labels, test_predictions)

        if epoch == n_epoch - 1:
            score_list = get_list_of_scores()
            for scr in score_list:
                best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, best_test_performance_dict[scr]))
            best_val_test_prediction_fl.write(best_test_predictions)

            best_val_test_result_fl.close()
            best_val_test_prediction_fl.close()


        

