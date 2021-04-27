# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:40:30 2021

@author: Colin
"""

import os
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from ConvFeatureModel import createFullSpeciesModel, createExperimentalFullSpeciesModel, createFlexibleConvFeatureModel
from collections import deque
import ConfigurationManager as cf
import time

class Agent():
    def __init__(self, empty_state, empty_extra, time_steps):
        self.time_steps = time_steps
        
        # allows random states to match random extras for training
        self.gi_que = deque()
        
        # lets all empty timesteps share the same memory
        self.empty_state = empty_state
        self.empty_extra = empty_extra
        
        # stores states observed by actor as ndarrays
        self.state_replay_memory = []
        
        # stores additional values from states as 
        # [ pxcor, pycor, patch-type of patch-ahead 1, heading, ticks, hunger, carried_wood, carried_stone, action, reward ]
        self.extras_replay_memory = []
    
    def add_state(self, new_state, new_extra):
        self.state_replay_memory[-1].append(new_state)
        self.extras_replay_memory[-1].append(new_extra)
    
    def start_generation(self, start_state, start_extra, to_delete):
        # fills start of each new generation with empty timesteps so there's always enough to predict on
        self.state_replay_memory.append([self.empty_state for __ in range(self.time_steps)]) 
        self.state_replay_memory[-1].append(start_state)
        self.extras_replay_memory.append([self.empty_extra for __ in range(self.time_steps)])
        self.extras_replay_memory[-1].append(start_extra)
        # deletes oldest state when SpeciesManager instructs it to
        if to_delete:
            del self.state_replay_memory[0]
            del self.extras_replay_memory[0]
    
    def get_training_state(self, rng):
        # calculates probability of each generation to get sampled from,
        # preference is towards sets where the agent survived longer
        probs = np.asarray([len(gen)
                            if len(gen) > self.time_steps * 2 else 0 
                            for i, gen in enumerate(self.extras_replay_memory)])
        probsum = sum(probs)
        probs = probs / probsum
        g_i = rng.choice(len(self.state_replay_memory), p = probs)
        s_i = rng.integers(len(self.state_replay_memory[g_i]) - (self.time_steps + 1))
        # sets up extras to be obtained properly and in the correct order
        self.gi_que.append([g_i, s_i])
        # returns list of length timesteps + 1, full of sequential ndarrays
        return self.state_replay_memory[g_i][s_i:s_i + self.time_steps + 1]
    
    def get_training_extra(self):
        # uses indexes given by random state
        g_i, s_i = self.gi_que.popleft()
        # returns list of extras of length timesteps + 1, full of sequential ndarrays
        return self.extras_replay_memory[g_i][s_i:s_i + self.time_steps + 1]
    
    def get_current_states(self):
        # returns the most recent states to predict on
        return self.state_replay_memory[-1][-self.time_steps:]
    
    def get_current_extras(self):
        # returns the most recent extras to predict on
        return [e[:-2] for e in self.extras_replay_memory[-1][-self.time_steps:]]

class SpeciesManager():
    """Manages Actors and their Neural Network.

    References:
       - For replay memory, use of a seperate target model, idea to clip rewards, and decision to use RMSprop:
           Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.
       - For general idea of reinforcement learning:
           https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html
       - Helped clarify Deep Q Network
           https://towardsdatascience.com/qrash-course-deep-q-networks-from-the-ground-up-1bbda41d3677

    """
    def __init__(self, species_names, num_actions, run_save_dir, channels=3, num_extras=10, min_world_val = 0, max_world_val = 25, saved_model=None):
        # Creating Required Directories in case they don't exist
        modelDirName = run_save_dir + '/models'
        if not os.path.exists(modelDirName):
            os.mkdir(modelDirName)
        self.dirName = modelDirName
        # dimensions of agent states
        dim = cf.AGENT_SIGHT_RADIUS * 2 + 1
        self.learning_rate = cf.LEARNING_RATE
        # allows for a saved model to be the initial model used in a training run
        # untested and no support outside of SpeciesManager, would need to change code in main to use
        # just add a model name to the construction in main, and potentialy adjust variables like exploration rate
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        if saved_model != None:
            model = tf.keras.models.load_model(saved_model)
            self.model = model
            target_model = tf.keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())
            self.target_model = target_model
        # creates fresh model and target
        else:
            model = createFullSpeciesModel(len(species_names), cf.TIMESTEPS_IN_OBSERVATIONS, dim, channels, num_extras - 2, num_actions, "fullModel", species_names)
            # model = createExperimentalFullSpeciesModel(len(species_names), cf.TIMESTEPS_IN_OBSERVATIONS, dim, channels, num_extras - 2, num_actions, "fullModel", species_names)
            model.compile(tf.keras.optimizers.Adam(learning_rate=self.learning_rate), "mse")
            self.model = model
            target_model = tf.keras.models.clone_model(model)
            target_model.set_weights(model.get_weights())
            self.target_model = target_model
        
        self.gen_training_time = 0
        self.gen_predicting_time = 0
        self.gen_data_time = 0
        self.gen_total_predictions = 0
        self.gen_total_training = 0
        self.dim = dim
        self.channels = channels
        self.num_extras = num_extras
        self.num_actions = num_actions
        self.average_action_sentiments = np.zeros((cf.NUM_SPECIES, num_actions),dtype=np.float64)
        self.time_steps = cf.TIMESTEPS_IN_OBSERVATIONS
        self.max_gens_in_mem = cf.MAX_GENS_IN_MEM
        self.radius = cf.AGENT_SIGHT_RADIUS
        self.min_world_val = min_world_val
        self.max_world_val = max_world_val
        self.num_actions = num_actions
        self.batch_size = cf.BATCH_SIZE
        self.discount_rate = cf.DISCOUNT_RATE
        self.rng = np.random.default_rng()

        # variables for memory and actor management
        self.mem_tracker = 0
        self.empty_state = np.zeros((dim, dim, channels), dtype=np.int8)
        self.empty_extra = np.zeros((num_extras,), dtype=np.float32)
        self.empty_current_state = [self.empty_state for __ in range(self.time_steps)]
        self.empty_current_extra = [self.empty_extra[:-2] for __ in range(self.time_steps)]
        self.actor_sets = [[Agent(self.empty_state, self.empty_extra, self.time_steps) 
                               for __ in range(cf.MAX_TOTAL_POP)] 
                                   for __ in range(len(species_names))]
        
        # variables for t-sne analysis later
        self.species0_prediction_values = []
        self.species0_prediction_input_states = []
        self.species0_prediction_input_extras = []
        self.num_tsne_inputs = 100_000
    
    def new_generation(self, actor_ids, start_states, start_extras):
        self.gen_training_time = 0
        self.gen_predicting_time = 0
        self.gen_data_time = 0
        self.gen_total_predictions = 0
        self.gen_total_training = 0
        self.average_action_sentiments *= 0
        if (self.mem_tracker > 0):
            for __ in range(int(cf.EPISODE_LENGTH / self.max_gens_in_mem)):
                self.train_from_memory()
        
        if self.mem_tracker < self.max_gens_in_mem:
            to_delete = False
            self.mem_tracker += 1
        else:
            to_delete = True
       
            
        
        for i, species in enumerate(actor_ids):
            for j, actor_id in enumerate(species):
                self.actor_sets[i][int(actor_id)].start_generation(start_states[i][j], start_extras[i][j], to_delete)      
        
        
     
    def add_to_actors(self, actor_ids, states, extras):
        self.mem_tracker += 1
        for i, species in enumerate(actor_ids):
            for j, actor_id in enumerate(species):
                self.actor_sets[i][int(actor_id)].add_state(states[i][j], extras[i][j])
     
    def decide_actions(self, living_ids):
        start_time = time.perf_counter()
        # this section allows size of input to be trimmed down while exploration rate is lower or many actors are dead
        # required due to parrallel nature of full species model needing each species input to be the same size
        # significantly speeds up the time required for predictions
        
        min_entries = max([len(s) for s in living_ids])
        
        if (min_entries < len(self.actor_sets[0])):
            
            actions = np.full((len(self.actor_sets),len(self.actor_sets[0])),-1)
            
            if (min_entries > 0):
                # this section makes all species have the same number of inputs to the neural net
                targets = [survivors.copy() for survivors in living_ids]
                for ix in range(len(targets)):
                    while len(targets[ix]) < min_entries:
                        targets[ix].append(-1)
                        
                states = [[self.empty_current_state if a == -1 else
                                      species[a].get_current_states() 
                                          for a in targets[sx]]
                                              for sx, species in enumerate(self.actor_sets)]
                
                extras = [[self.empty_current_extra if a == -1 else
                                      species[a].get_current_extras()
                                          for a in targets[sx]]
                                              for sx, species in enumerate(self.actor_sets)]
    
                # Normalizes the world data to be a float between 0 and 1
                # done here so that when stored, ndarray can save space by being np.int8
                inputs = [[( np.asarray(states[i]).astype(np.float32) - self.min_world_val ) / 
                                   ( self.max_world_val - self.min_world_val ),
                                       np.asarray(extras[i])] for i in range(len(states))]

                self.gen_data_time += time.perf_counter() - start_time
                start_time = time.perf_counter()
                
                best_actions = np.argmax(self.model(inputs, training=False), axis=2)
                self.gen_predicting_time += time.perf_counter() - start_time
                start_time = time.perf_counter()
                
                # ensures only the predictions on actual states are passed to the actors
                for i in range(len(best_actions)):
                    for idx, idnum in enumerate(living_ids[i]):
                        actions[i,idnum] = best_actions[i,idx]
        
        # runs if all actors of any species need predictions    
        else:      
            states = np.asarray([[a.get_current_states() for a in species] for species in self.actor_sets])
            extras = np.asarray([[a.get_current_extras() for a in species] for species in self.actor_sets])
            inputs = [[states[i], extras[i]] for i in range(len(states))]
            self.gen_data_time += time.perf_counter() - start_time
            start_time = time.perf_counter()
            actions = np.argmax(self.model(inputs, training=False), axis=2)
            self.gen_predicting_time += time.perf_counter() - start_time
            start_time = time.perf_counter()
        
        self.gen_data_time += time.perf_counter() - start_time
        self.gen_total_predictions += min_entries        
        return actions
    
    
    def train_from_memory(self):
        start_time = time.perf_counter()
        # generates random sample of actors for each species
        actor_samples = [self.rng.integers(len(species), size=self.batch_size) for species in self.actor_sets]
        
        # grabs states from selected actors for each species
        states = [[self.actor_sets[s][a].get_training_state(self.rng)
                                  for a in a_ix] for s, a_ix in enumerate(actor_samples)]
        
        self.species0_prediction_input_states.extend(s[-2] for s in states[0])
        
        # normalizes states
        states = ((np.asarray(states) - self.min_world_val) / (self.max_world_val - self.min_world_val)).astype(np.float32)
        
        # grabs extras from selected actors for each species
        extras = [[self.actor_sets[s][a].get_training_extra() for a in a_ix]
                                 for s, a_ix in enumerate(actor_samples)]
        
        self.species0_prediction_input_extras.extend(e[-2][:-2] for e in extras[0])
        
        extras = np.asarray(extras)

        # seperates the actions and rewards from the states
        actions = extras[:,:,-1,-2].astype(np.int8)
        # clips the rewards to account for any situations where the reward values multiply too high
        rewards = extras[:,:,-1,-1].clip(-1,1)
        
        # splits the timestep input being trained from the future input being used to get future reward values
        target_inputs = [[states[i,:,:-1], extras[i,:,:-1,:-2]] for i in range(len(states))]
        future_inputs = [[states[i,:,1:], extras[i,:,1:,:-2]] for i in range(len(states))]
        self.gen_data_time += time.perf_counter() - start_time
        start_time = time.perf_counter()
        # gets future rewards
        future_rewards = np.max(self.target_model(future_inputs, training=False), axis = 2)
        # gets target prediction values for training
        training_targets = np.asarray(self.model(target_inputs, training=False))
        
        self.gen_total_predictions += (self.batch_size * 2)
        self.gen_predicting_time += time.perf_counter() - start_time
        start_time = time.perf_counter()
        
        self.average_action_sentiments += training_targets.sum(axis=1)
        
        self.species0_prediction_values.extend(training_targets[0])
        if (len(self.species0_prediction_values) > self.num_tsne_inputs):
            del self.species0_prediction_values[:self.batch_size]
            del self.species0_prediction_input_states[:self.batch_size]
            del self.species0_prediction_input_extras[:self.batch_size]
        
        # pre_adjustment_values = training_targets.copy().reshape(-1,13)
        
        # uses the Deep Q Network approximation of the Bellman Equeation to update targets
        for species in range(len(training_targets)):
            for index in range(len(training_targets[species])):
                if actions[species,index] > -1:
                    training_targets[species,index,actions[species,index]] = \
                        rewards[species,index] + self.discount_rate * future_rewards[species,index]
                elif actions[species,index] == -1:
                    training_targets[species,index,:] = rewards[species,index]
                else:
                    # on death, penalize every action except for eating
                    training_targets[species,index,:6] = rewards[species,index]
                    training_targets[species,index,7:] = rewards[species,index]
        
        # prediction_strings = ["move_up","move_lt","move_rt","move_dn","turn_rt","turn_lt","eat","cut","mine","build_w","build_s","build_b","destroy"]
        # flat_rews = rewards.flatten()
        # flat_acts = actions.flatten()
        # flat_futures = future_rewards.flatten()
        # flat_newvals = training_targets.reshape(-1,13)
 
        target_list = [a for a in training_targets]
        self.gen_data_time += time.perf_counter() - start_time
        # trains model with the inputs to training targets and
        # the updated predictions from those inputs
        start_time = time.perf_counter()
        self.model.train_on_batch(target_inputs, target_list)
        # new_values = np.asarray(self.model(target_inputs, training=False)).reshape(-1,13)
        
        # print("------------------------------------------------------------------------")
        # print("|  action  | init value |   reward   | future rew | target val | new value  |   change   |")
        # for ex in range(len(pre_adjustment_values)):
        #     print("| {:^8} | {:10.6f} | {:10.6f} | {:10.6f} | {:10.6f} | {:10.6f} | {:10.6f} |".format(prediction_strings[flat_acts[ex]],
        #                                                                                                pre_adjustment_values[ex,flat_acts[ex]],
        #                                                                                                flat_rews[ex],flat_futures[ex],
        #                                                                                                flat_newvals[ex,flat_acts[ex]],
        #                                                                                                new_values[ex,flat_acts[ex]],
        #                                                                                                new_values[ex,flat_acts[ex]] - pre_adjustment_values[ex,flat_acts[ex]]))
        # print("------------------------------------------------------------------------")
        
        
        self.gen_training_time += time.perf_counter() - start_time
        self.gen_total_training += self.batch_size
    
    def update_target(self):
        # updates target model weights to current model weights when called from main
        self.target_model.set_weights(self.model.get_weights())
        
        
    def increase_learning(self):
        self.learning_rate = self.learning_rate * cf.FAIL_STREAK_LR_MOD
        # used answer by Tomé Silva here: https://stackoverflow.com/questions/59737875/keras-change-learning-rate
        self.model.optimizer.learning_rate.assign(self.learning_rate)
        
    def decrease_learning(self):
        self.learning_rate = self.learning_rate * cf.SUCCESS_STREAK_LR_MOD
        # used answer by Tomé Silva here: https://stackoverflow.com/questions/59737875/keras-change-learning-rate
        self.model.optimizer.learning_rate.assign(self.learning_rate)
    
    def get_learning_rate(self):
        return self.learning_rate
    
    def get_gen_timings(self):
        return self.gen_training_time, self.gen_total_training, self.gen_predicting_time, self.gen_total_predictions, \
               (self.average_action_sentiments / self.gen_total_training), self.gen_data_time
               
    def get_tsne_data(self):
        return self.species0_prediction_values, self.species0_prediction_input_states, self.species0_prediction_input_extras

    def savemodel(self, savetag):
        self.model.save(self.dirName + "/fullModel_gen" + savetag + ".h5", overwrite=True)
        
    def getSpeciesModel(self):
        return createFlexibleConvFeatureModel(cf.TIMESTEPS_IN_OBSERVATIONS, self.dim, self.channels, self.num_extras - 2, self.num_actions, "ExampleSpeciesModel")
        
# takes a trained model to create, same idea as normal manager just simplified to not train
# meant to be used by showcase to run saved models and show results
class TrainedSpeciesManager():
    def __init__(self, model_name, species_names, num_actions, channels=3, num_extras=10, min_world_val = 0, max_world_val = 25):
        dim = cf.AGENT_SIGHT_RADIUS * 2 + 1
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')
        self.model = tf.keras.models.load_model(model_name)
        self.time_steps = cf.TIMESTEPS_IN_OBSERVATIONS
        self.max_gens_in_mem = cf.MAX_GENS_IN_MEM
        self.radius = cf.AGENT_SIGHT_RADIUS
        self.min_world_val = min_world_val
        self.max_world_val = max_world_val
        self.rng = np.random.default_rng()

        self.mem_tracker = 0
        self.empty_state = np.zeros((dim, dim, channels), dtype=np.int8)
        self.empty_current_state = [self.empty_state for __ in range(self.time_steps)]
        self.empty_extra = np.zeros((num_extras,), dtype=np.float32)
        self.empty_current_extra = [self.empty_extra[:-2] for __ in range(self.time_steps)]
        self.actor_sets = [[Agent(self.empty_state, self.empty_extra, self.time_steps) 
                               for __ in range(cf.SPECIES_START_POP)] 
                                   for __ in range(len(species_names))]
    
    
    def new_generation(self, actor_ids, start_states, start_extras):
        self.mem_tracker += 1
        for i, species in enumerate(actor_ids):
            for j, actor_id in enumerate(species):
                self.actor_sets[i][int(actor_id)].start_generation(start_states[i][j], start_extras[i][j], False)      
    
    def add_to_actors(self, actor_ids, states, extras):
        self.mem_tracker += 1
        for i, species in enumerate(actor_ids):
            for j, actor_id in enumerate(species):
                self.actor_sets[i][int(actor_id)].add_state(states[i][j], extras[i][j])      
     
    def decide_actions(self): 
        states = np.asarray([[a.get_current_states() for a in species] for species in self.actor_sets])
        extras = np.asarray([[a.get_current_extras() for a in species] for species in self.actor_sets])
        inputs = [[states[i], extras[i]] for i in range(len(states))]
        actions = np.argmax(self.model(inputs, training = False), axis=2)
        return actions