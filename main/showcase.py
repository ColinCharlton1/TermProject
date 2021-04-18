# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:56:02 2021

@author: Colin
"""

import os
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import pyNetLogo
from AgentManagement import TrainedSpeciesManager
from WorldUtils import update_world, get_world_data, get_masked_world_data, size_string, get_species_dist_string


# Breaks Everything if Changed
NUM_ACTIONS = 13 # must be 13
WORLD_FEATURES = 3 # must be 3

# Not Recomended to Change
AGENT_SIGHT_RADIUS = 9
TIMESTEPS_IN_OBSERVATIONS = 5
WORLD_HEIGHT = 75
WORLD_WIDTH = 120
PATCH_SIZE = 10

# Feel Free to Change
NUM_SPECIES = 6 # max 10
SPECIES_START_POP = 30
NUM_ISLANDS = 10
MAX_TOTAL_POP = SPECIES_START_POP * NUM_ISLANDS
LEARNING_RATE = 0.0008
MAX_MEM_SIZE = 20000
BATCH_SIZE = 512
TRAIN_FREQUENCY = 10
UPDATE_TARGET_FREQUENCY = 200
DISCOUNT_RATE = 0.99
EPISODE_LENGTH = 500
NUMBER_OF_GENERATIONS = 5000
EXPLORATION_DECAY = 0.0008
NUM_PROCESSES = 5
MIN_ALIVE = 5
EXPLORATION_METHOD = "solo" # either solo or shared
MASK_VISION = False
    
def run_showcase(startgen, endgen):
    # Variable Initialization
    team_species = int(NUM_SPECIES/2)
    species_list = ["red1", "red2", "red3", "red4", "red5"][:team_species] + \
                   ["purple1", "purple2", "purple3", "purple4", "purple5"][:team_species]
                
    # must change if more than 10 islands used
    # format [rock_adjustment, tree_adjustment, bush_adjustment, puddle_adjustment, river_toggle]
    world_types = ["8 3 3 1 1",
                   "6 3 3 3 1",
                   "6 2 2 1 1",
                   "6 3 3 3 1",
                   "6 2 2 1 1",
                   "3 1 1 1 1",
                   "2 2 2 1 1",
                   "8 0 0 1 1",
                   "8 -2 -2 1 1",
                   "4 1 1 -2 1",
                   "3 1 1 -1 1",
                   "-5 1 5 1 1",
                   "-10 1 1 1 1",
                   "4 1 1 5 0",
                   "-10 3 3 10 0"] 
    
    
    population_dist = np.zeros((NUM_SPECIES)).astype(np.int32) + SPECIES_START_POP

    netlogo = pyNetLogo.NetLogoLink(gui = True)
    netlogo.load_model("C:/Users/Colin/Desktop/Winter2021/CPSC565/TermProject/repo/TeamBuilderSim.nlogo")
    
    world_size = size_string(WORLD_WIDTH, WORLD_HEIGHT, PATCH_SIZE)
    team_pop = " " + str(SPECIES_START_POP * NUM_SPECIES / 2)
    
    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros/46115998 answer from MSeifert used to make padded array
    padding = 2 * AGENT_SIGHT_RADIUS
    padded_world = np.zeros((WORLD_HEIGHT + padding, WORLD_WIDTH + padding, WORLD_FEATURES)).astype(np.int8) 
    
    for gen in range(startgen, endgen + 1, 5):
        netlogo.command("setup " + get_species_dist_string(population_dist))
        netlogo.command("set-env-params " + world_types[0] + world_size + team_pop)
        netlogo.command("setup-environment")
        
        # Species Manager Setup (The Neural Network and Memory Manager)
        species_manager = TrainedSpeciesManager("C:/Users/Colin/Desktop/Winter2021/CPSC565/TermProject/repo/main/models/fullModel" + str(gen) + ".h5",
                                                species_list,
                                                SPECIES_START_POP,
                                                TIMESTEPS_IN_OBSERVATIONS,
                                                AGENT_SIGHT_RADIUS,
                                                MAX_MEM_SIZE)          
        
        # first NUM_SPECIES + 1 values in pro_ids are used to partition ids into proper species
        vegetation = netlogo.report("env-vegetation")
        actors = netlogo.report("env-actors")
        color = netlogo.report("env-types")
        new_world = np.column_stack((vegetation, actors, color)).clip(-25, 25).astype(np.int8)

        new_world = np.flipud(new_world.reshape(WORLD_HEIGHT, WORLD_WIDTH, WORLD_FEATURES))

        padded_world[AGENT_SIGHT_RADIUS:new_world.shape[0] + AGENT_SIGHT_RADIUS,
                     AGENT_SIGHT_RADIUS:new_world.shape[1] + AGENT_SIGHT_RADIUS] = new_world.copy()
        
        current_world = padded_world.copy()
        
        
        i = 1
        ########################################### tick loop ############################################
        while (i < EPISODE_LENGTH + 1):
            living_ids = netlogo.report("get-ids")
            if not type(living_ids) is np.ndarray:
                break       
            living_ids = living_ids.astype(np.int32)
            world_updates = netlogo.report("get-updates")
            new_species_vals = netlogo.report("get-actor-data")
            

            species_ids = [[] for __ in range(NUM_SPECIES)]
            all_new_states = [[] for __ in range(NUM_SPECIES)]
            all_new_extras = [[] for __ in range(NUM_SPECIES)]
            # first 6 values in pro_ids are used to partition ids into proper species
            offset = living_ids[0]
            
            if type(world_updates) is np.ndarray:
                update_world(current_world, world_updates.astype(np.int16), AGENT_SIGHT_RADIUS)

            if MASK_VISION:
                new_states = get_masked_world_data(current_world, new_species_vals, AGENT_SIGHT_RADIUS)
            else:
                new_states = get_world_data(current_world, new_species_vals, AGENT_SIGHT_RADIUS)
                
            new_species_vals[:,0] = new_species_vals[:,0] / WORLD_WIDTH
            new_species_vals[:,1] = new_species_vals[:,1] / WORLD_HEIGHT
            
            start = 0
            for s in range(NUM_SPECIES):
                end = start + living_ids[s + 1]
                species_ids[s].extend(living_ids[start + offset: end + offset] - population_dist[:s].sum())
                all_new_states[s].extend(new_states[start:end])
                all_new_extras[s].extend(new_species_vals[start:end])
                start = end  
    
            if i == 1:
                species_manager.new_generation(all_new_states, all_new_extras, [current_world])
                print("generation started")
            else:
                species_manager.add_to_actors(
                    species_ids, all_new_states, all_new_extras, [current_world])
                
            ############################## get best actions ############################
            actions = np.zeros(len(living_ids[offset:]))
            bestActions = species_manager.decide_actions().flatten()
            # print("best actions: ", bestActions)
            # print("living ids: ", living_ids[offset:])
            for index, idnum in enumerate(living_ids[offset:]):
                actions[index] = bestActions[idnum]
            
            ############################## setting actions ##############################       
            df = pd.DataFrame(np.stack([living_ids[offset:], actions], axis=1), columns=["who","action"])
            netlogo.write_NetLogo_attriblist(df[["who","action"]], "actor")
            time.sleep(1)
            netlogo.command("end-step")
            i += 1
        

    netlogo.kill_workspace() # Killed cause unnecessary
    print("ready to be killed")


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logging.getLogger("tensorflow").setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    run_showcase(70,80)
    
    