# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:56:02 2021

@author: Colin
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pyNetLogo
from AgentManagement import TrainedSpeciesManager
import ConfigurationManager as cf
from WorldUtils import update_world, get_world_data, get_masked_world_data, get_species_dist_string


# Breaks Everything if Changed
NUM_ACTIONS = 13 # must be 13
WORLD_FEATURES = 3 # must be 3

TARGET_RUN_NAME = "18_03_23"
START_GENERATION = 5
END_GENERATION = 30
    
def run_showcase():
    # Tensorflow Settings
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    np.set_printoptions(threshold=1000000, linewidth=1000000)
    
    # Creating Required Directories in case they don't exist
    modelDirName = 'models'
    if not os.path.exists(modelDirName):
        os.mkdir(modelDirName)
    
    
    # Useful Variable Initialization
    exploration_rates = get_exploration_rates()
    team_species = int(cf.NUM_SPECIES/2)
    species_list = ["red1", "red2", "red3", "red4", "red5"][:team_species] + \
                   ["purple1", "purple2", "purple3", "purple4", "purple5"][:team_species]
    
    
    myrng = np.random.default_rng()
    population_dist = np.zeros((cf.NUM_SPECIES)).astype(np.int32) + cf.SPECIES_START_POP
    
    netLogo_model_path = os.path.abspath("TeamBuilderSim.nlogo")
    netlogo = pyNetLogo.NetLogoLink(gui = True)
    netlogo.load_model(netLogo_model_path)
    
    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros/46115998 
    # answer from MSeifert used to make padded array for world management
    padding = 2 * cf.AGENT_SIGHT_RADIUS
    padded_world = np.zeros((cf.WORLD_HEIGHT + padding, cf.WORLD_WIDTH + padding, WORLD_FEATURES)).astype(np.int8) 
    
    for gen in range(START_GENERATION, END_GENERATION + 1, 5):
        print("model from generation {}:".format(gen))
        netlogo.command("set-env-params " + cf.get_world_configs(0))
        netlogo.command("set-reward-params" + cf.get_reward_configs())
        netlogo.command("setup " + get_species_dist_string(population_dist))
        
        # Species Manager Setup (The Neural Network and Memory Manager)
        model_name = os.getcwd() + "/models/fullModel_" + TARGET_RUN_NAME + "_gen" + str(gen) + ".h5"
        species_manager = TrainedSpeciesManager(model_name,
                                                species_list,
                                                NUM_ACTIONS)
        
        
        # first NUM_SPECIES + 1 values in pro_ids are used to partition ids into proper species
        vegetation = netlogo.report("env-vegetation")
        actors = netlogo.report("env-actors")
        color = netlogo.report("env-types")
        new_world = np.column_stack((vegetation, actors, color)).clip(-25, 25).astype(np.int8)


        new_world = np.flipud(new_world.reshape(cf.WORLD_HEIGHT, cf.WORLD_WIDTH, WORLD_FEATURES))

        padded_world[cf.AGENT_SIGHT_RADIUS:new_world.shape[0] + cf.AGENT_SIGHT_RADIUS,
                     cf.AGENT_SIGHT_RADIUS:new_world.shape[1] + cf.AGENT_SIGHT_RADIUS] = new_world.copy()
            
        current_world = padded_world.copy()
        exploration_rate = exploration_rates[gen]
        i = 1
        ########################################### tick loop ############################################
        while i < cf.EPISODE_LENGTH + 2:
            living_ids = netlogo.report("get-ids")
            if not type(living_ids) is np.ndarray:
                break       
            living_ids = living_ids.astype(np.int32)
            world_updates = netlogo.report("get-updates")
            new_species_vals = netlogo.report("get-actor-data")
            

            species_ids = [[] for __ in range(cf.NUM_SPECIES)]
            all_new_states = [[] for __ in range(cf.NUM_SPECIES)]
            all_new_extras = [[] for __ in range(cf.NUM_SPECIES)]
            # first 6 values in pro_ids are used to partition ids into proper species
            offset = living_ids[0]
            
            if type(world_updates) is np.ndarray:
                update_world(current_world, world_updates.astype(np.int16), cf.AGENT_SIGHT_RADIUS)

            if cf.MASK_VISION:
                new_states = get_masked_world_data(current_world, new_species_vals, cf.AGENT_SIGHT_RADIUS)
            else:
                new_states = get_world_data(current_world, new_species_vals, cf.AGENT_SIGHT_RADIUS)
                
                
            new_species_vals[:,0] = new_species_vals[:,0] / cf.WORLD_WIDTH
            new_species_vals[:,1] = new_species_vals[:,1] / cf.WORLD_HEIGHT
            
            start = 0
            for s in range(cf.NUM_SPECIES):
                end = start + living_ids[s + 1]
                species_ids[s].extend(living_ids[start + offset: end + offset] - population_dist[:s].sum())
                all_new_states[s].extend(new_states[start:end])
                all_new_extras[s].extend(new_species_vals[start:end])
                start = end  
    
            if i == 1:
                species_manager.new_generation(all_new_states, all_new_extras)
            else:
                species_manager.add_to_actors(species_ids, all_new_states, all_new_extras)

            ############################## get best actions ############################
            actions = myrng.integers(0, NUM_ACTIONS, size=len(living_ids[offset:]))
            bestActions = species_manager.decide_actions().flatten()
            # print("best actions: ", bestActions)
            # print("living ids: ", living_ids[offset:])
            for index, idnum in enumerate(living_ids[offset:]):
                if myrng.random() > exploration_rate:
                    actions[index] = bestActions[idnum]
            
            ############################## setting actions ##############################       
            df = pd.DataFrame(np.stack([living_ids[offset:], actions], axis=1), columns=["who","action"])
            netlogo.write_NetLogo_attriblist(df[["who","action"]], "actor")
            netlogo.command("end-step")
            i += 1
        

    netlogo.kill_workspace() # Killed cause unnecessary
    print("ready to be killed")


def get_exploration_rates():
    target = "rawdata/exploration_history" + TARGET_RUN_NAME + ".csv"
    rates = np.loadtxt(target).flatten()
    return rates

if __name__ == "__main__":
    run_showcase()
    
    