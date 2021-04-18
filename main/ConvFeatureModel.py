# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:01:18 2021

@author: Colin
"""

import os
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# general layout for convolution cobbled together from the following:
# https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/
# https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
# https://deeplizard.com/learn/video/ZjM_XQa5s6s
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
# way to have seperate inputs found at:
# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
def createConvFeatureModelConv1D(timesteps, length, channels, n_extras, n_actions, name):
    inputs1 = tf.keras.Input(shape=(timesteps,length,length,channels))
    inputs2 = tf.keras.Input(shape=(timesteps,n_extras))
    env_features = tf.keras.layers.Conv3D( filters=64, kernel_size=(2,3,3), padding="valid" , activation="relu" )(inputs1)
    env_features = tf.keras.layers.Conv3D( filters=64, kernel_size=(2,2,2), padding="valid" , activation="relu" )(env_features)
    env_features = tf.keras.layers.MaxPool3D(pool_size=(1,2,2))(env_features)
    env_features = tf.keras.layers.Conv3D( filters=64, kernel_size=(2,4,4), padding="valid" , activation="relu" )(env_features)
    env_features = tf.keras.layers.Flatten()(env_features)
    
    extra_features = tf.keras.layers.Conv1D(filters=16, kernel_size=2, padding="valid" , activation="relu")(inputs2)
    extra_features = tf.keras.layers.Conv1D(filters=16, kernel_size=2, padding="valid" , activation="relu")(extra_features)
    extra_features = tf.keras.layers.MaxPool1D(pool_size=2)(extra_features)
    extra_features = tf.keras.layers.Flatten()(extra_features)
    
    x = tf.keras.layers.Concatenate(axis=1)([env_features, extra_features])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_actions)(x)
    model = tf.keras.Model(inputs = [inputs1, inputs2], outputs = outputs, name=name)
    return model

# idea for running models in parrallel this way came from here, answer from Daniel Möller
# https://stackoverflow.com/questions/50474336/how-to-run-multiple-keras-programs-on-single-gpu
def createFullSpeciesModel(num_species, timesteps, length, channels, n_extras, n_actions, name, species_names):
    inputs = []
    outputs = []
    models = []
    for i in range(num_species):
        species_model = createConvFeatureModelConv1D(timesteps, length, channels, n_extras, n_actions, species_names[i])
        models.append(species_model)
        inputs.append(species_model.input)
        outputs.append(models[i](inputs[i]))
    model = tf.keras.Model(inputs=inputs,outputs=outputs, name=name)
    return model
