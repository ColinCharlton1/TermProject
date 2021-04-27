# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:01:18 2021

@author: Colin
"""
# import os
# import logging
# logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import ConfigurationManager as cf

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
    x = tf.keras.layers.Dropout(cf.DROPOUT_RATE)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_actions, dtype='float32')(x)
    model = tf.keras.Model(inputs = [inputs1, inputs2], outputs = outputs, name=name)
    return model

# general layout for convolution cobbled together from the following:
# https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/
# https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
# https://deeplizard.com/learn/video/ZjM_XQa5s6s
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
# way to have seperate inputs found at:
# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
def createFlexibleConvFeatureModel(timesteps, length, channels, n_extras, n_actions, name):
    inputs1 = tf.keras.Input(shape=(timesteps,length,length,channels))
    env_features = inputs1
    for filters, kernel in cf.NETWORK_CONV_ARCHITECTURE:
        
        if filters != 0:
            env_features = tf.keras.layers.Conv3D(filters=filters,
                                                  kernel_size=kernel,
                                                  padding="valid" , activation="relu" )(env_features)
            
        else:
            env_features = tf.keras.layers.MaxPool3D(pool_size=kernel)(env_features)
            
    env_features = tf.keras.layers.Flatten()(env_features)
    
    inputs2 = tf.keras.Input(shape=(timesteps,n_extras))
    extra_features = inputs2
    for filters, kernel in cf.EXTRA_CONV_LAYERS:
        if filters == 0:
            extra_features = tf.keras.layers.MaxPool1D(pool_size=kernel)(extra_features)
        else:
            extra_features = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel, padding="valid" , activation="relu")(extra_features)
    
    extra_features = tf.keras.layers.Flatten()(extra_features)
    
    x = tf.keras.layers.Concatenate(axis=1)([env_features, extra_features])
    for dl in cf.DENSE_LAYERS:
        x = tf.keras.layers.Dense(dl, activation="relu")(x)
        
    outputs = tf.keras.layers.Dense(n_actions, dtype='float32')(x)
    model = tf.keras.Model(inputs = [inputs1, inputs2], outputs = outputs, name=name)
    return model

# idea for running models in parrallel this way came from here, answer from Daniel Möller
# https://stackoverflow.com/questions/50474336/how-to-run-multiple-keras-programs-on-single-gpu
def createFullSpeciesModel(num_species, timesteps, length, channels, n_extras, n_actions, name, species_names):
    inputs = []
    outputs = []
    models = []
    for i in range(num_species):
        species_model = createFlexibleConvFeatureModel(timesteps, length, channels, n_extras, n_actions, species_names[i])
        models.append(species_model)
        inputs.append(species_model.input)
        outputs.append(models[i](inputs[i]))
    model = tf.keras.Model(inputs=inputs,outputs=outputs, name=name)
    return model

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # createExperimentalConvFeatureModel
    # model = createExperimentalConvFeatureModel(cf.TIMESTEPS_IN_OBSERVATIONS, cf.AGENT_SIGHT_RADIUS * 2 + 1, 3, 7, 13, "test")
    # model = createFlexibleConvFeatureModel(cf.TIMESTEPS_IN_OBSERVATIONS, cf.AGENT_SIGHT_RADIUS * 2 + 1, 3, 7, 13, "test")
    # model.compile(tf.optimizers.RMSprop(learning_rate=0.001), "mse")
    # model.summary()
    
    
    
    
# general layout for convolution cobbled together from the following:
# https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/
# https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
# https://deeplizard.com/learn/video/ZjM_XQa5s6s
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
# way to have seperate inputs found at:
# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
def createExperimentalConvFeatureModel(timesteps, length, channels, n_extras, n_actions, name):
    inputs1 = tf.keras.Input(shape=(timesteps,length,length,channels))
    inputs2 = tf.keras.Input(shape=(timesteps,n_extras))
    env_features = tf.keras.layers.Conv3D( filters=32, kernel_size=(3,4,4), padding="valid" , activation="relu")(inputs1)
    env_features = tf.keras.layers.MaxPool3D(pool_size=(1,2,2))(env_features)
    env_features = tf.keras.layers.Conv3D( filters=40, kernel_size=(3,3,3), padding="valid" , activation="relu")(env_features)
    env_features = tf.keras.layers.MaxPool3D(pool_size=(2,3,3))(env_features)
    env_features = tf.keras.layers.Conv3D( filters=48, kernel_size=(3,2,2), padding="valid" , activation="relu")(env_features)
    env_features = tf.keras.layers.Flatten()(env_features)
    
    local_area = tf.keras.layers.Cropping3D((0,13,13))(inputs1)
    local_area = tf.keras.layers.Conv3D( filters=16, kernel_size=(3,1,3), padding="valid" , activation="relu")(local_area)
    local_area = tf.keras.layers.Conv3D( filters=16, kernel_size=(3,3,1), padding="valid" , activation="relu")(local_area)
    local_area = tf.keras.layers.MaxPool3D(pool_size=(2,1,1))(local_area)
    local_area = tf.keras.layers.Conv3D( filters=16, kernel_size=(3,1,1), padding="valid" , activation="relu")(local_area)
    local_area = tf.keras.layers.Flatten()(local_area)
    
    extra_features = tf.keras.layers.Conv1D(filters=16, kernel_size=2, padding="valid" , activation="relu")(inputs2)
    extra_features = tf.keras.layers.Conv1D(filters=16, kernel_size=2, padding="valid" , activation="relu")(extra_features)
    extra_features = tf.keras.layers.MaxPool1D(pool_size=2)(extra_features)
    extra_features = tf.keras.layers.Flatten()(extra_features)
    
    x = tf.keras.layers.Concatenate(axis=1)([env_features, extra_features, local_area])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_actions)(x)
    model = tf.keras.Model(inputs = [inputs1, inputs2], outputs = outputs, name=name)
    return model



# idea for running models in parrallel this way came from here, answer from Daniel Möller
# https://stackoverflow.com/questions/50474336/how-to-run-multiple-keras-programs-on-single-gpu
def createExperimentalFullSpeciesModel(num_species, timesteps, length, channels, n_extras, n_actions, name, species_names):
    inputs = []
    outputs = []
    models = []
    for i in range(num_species):
        species_model = createExperimentalConvFeatureModel(timesteps, length, channels, n_extras, n_actions, species_names[i])
        models.append(species_model)
        inputs.append(species_model.input)
        outputs.append(models[i](inputs[i]))
    model = tf.keras.Model(inputs=inputs,outputs=outputs, name=name)
    return model