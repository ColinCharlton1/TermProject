# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:12:54 2021

@author: Colin
"""

import os
import logging
logging.getLogger("tensorflow").setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
from multiprocessing import Queue
import numpy as np
import pyNetLogo
from AgentManagement import SpeciesManager
from NetLogoProcess import NetLogoProcessManager
from DataManager import DataManager
import ConfigurationManager as cf
from WorldUtils import update_world, get_world_data, get_masked_world_data, distribute_populations, apply_exploration_rewards
from signal import signal, SIGINT
import sys
    
# Breaks Everything if Changed
NUM_ACTIONS = 13 # must be 13
WORLD_FEATURES = 3 # must be 3

def main():
    run_name = time.strftime("%Y-%m-%d-%H%M")
    run_save_dir = cf.SAVE_DIRECTORY + run_name
    if not os.path.exists(run_save_dir):
        os.mkdir(run_save_dir)
    # Tensorflow Settings
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    np.set_printoptions(threshold=1000000, linewidth=1000000, precision=6, suppress=True)
    
    # Useful Variable Initialization
    current_worlds = [0] * cf.NUM_ISLANDS
    exploration_rate = 1.0
    exploration_decay = cf.EXPLORATION_DECAY
    # fail_streak = 0
    # success_streak = 0
    team_species = int(cf.NUM_SPECIES/2)
    species_list = ["red1", "red2", "red3", "red4", "red5"][:team_species] + \
                   ["purple1", "purple2", "purple3", "purple4", "purple5"][:team_species]
     
    # best_gen_total_rewards = np.repeat(-100,(cf.NUM_ISLANDS * cf.NUM_SPECIES)).reshape((cf.NUM_ISLANDS, cf.NUM_SPECIES))
    myrng = np.random.default_rng()
    population_dist_probs = np.zeros((cf.NUM_SPECIES, cf.NUM_ISLANDS)).astype(np.float32) + 1 / cf.NUM_ISLANDS
    population_dist = np.zeros((cf.NUM_ISLANDS, cf.NUM_SPECIES)).astype(np.int32) + cf.SPECIES_START_POP
    team1_exploration_maps = np.zeros((cf.NUM_ISLANDS, cf.WORLD_WIDTH, cf.WORLD_HEIGHT, 4), dtype=np.int64)
    team2_exploration_maps = np.zeros((cf.NUM_ISLANDS, cf.WORLD_WIDTH, cf.WORLD_HEIGHT, 4), dtype=np.int64)
  
    # https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros/46115998 
    # answer from MSeifert used to make padded array for world management
    padding = 2 * cf.AGENT_SIGHT_RADIUS
    padded_world = np.zeros((cf.WORLD_HEIGHT + padding, cf.WORLD_WIDTH + padding, WORLD_FEATURES)).astype(np.int8) 

    # Recording Variable Initialization
    data_manager = DataManager(species_list, run_save_dir)
    
    # Species Manager Setup (The Neural Network and Memory Manager)
    species_manager = SpeciesManager(species_list, NUM_ACTIONS, run_save_dir)
    training_counter = 0
    
    # this is needed to start Java Virtual Machine so processes get attached to it
    mainNetLogo = pyNetLogo.NetLogoLink(gui=False)

    # Initializing NetLogoProcessManager class starts all the processes
    outque = Queue()
    netLogoProcessManager = NetLogoProcessManager(population_dist, outque, run_save_dir)
    
    
    g = 0
    # https://www.devdungeon.com/content/python-catch-sigint-ctrl-c
    def handler(signal_received, frame):
        # Handle any cleanup here
        print('SIGINT or CTRL-C detected. Exiting gracefully')
        data_manager.create_metadata(g, species_manager.getSpeciesModel())
        pred_vals, pred_in_states, pred_in_extras = species_manager.get_tsne_data()
        data_manager.commit_tsne_data(pred_vals, pred_in_states, pred_in_extras)
        netLogoProcessManager.kill_all()
        time.sleep(5)
        print("killing JVM...")
        mainNetLogo.kill_workspace()
        time.sleep(3)
        print("attempting to exit...")
        os._exit(1)
        sys.exit("exit complete")
        exit()
        quit()
        print("exit has catostrophically failed, why am I so bad at this?")
        
    signal(SIGINT, handler)
   
    ################################################## generation loop ##################################################
    while g < cf.NUMBER_OF_GENERATIONS:
        # start_snapshot = tracemalloc.take_snapshot()
        gen_start_time = time.time()
        gen_total_rewards = np.zeros((cf.NUM_ISLANDS, cf.NUM_SPECIES))
        action_frequency = np.zeros((cf.NUM_SPECIES, NUM_ACTIONS),dtype=np.int32)
        action_rewards = np.zeros((cf.NUM_SPECIES, NUM_ACTIONS),dtype=np.float64)
        max_action_rewards = np.zeros((cf.NUM_SPECIES, NUM_ACTIONS),dtype=np.float64)
        actions_taken = np.zeros((cf.NUM_SPECIES, NUM_ACTIONS),dtype=np.float64)
        red_advantage = np.zeros((cf.EPISODE_LENGTH + 2))
        purple_advantage = np.zeros((cf.EPISODE_LENGTH + 2))
        dead_islands = 0
        
        netLogoProcessManager.get_initial_worlds()

        # getting the new worlds initial states
        for n in range(cf.NUM_ISLANDS):
            # first NUM_SPECIES + 1 values in pro_ids are used to partition ids into proper species
            pro_id, new_world = outque.get()
            new_world = np.flipud(new_world.reshape(cf.WORLD_HEIGHT, cf.WORLD_WIDTH, WORLD_FEATURES))

            padded_world[cf.AGENT_SIGHT_RADIUS:new_world.shape[0] + cf.AGENT_SIGHT_RADIUS,
                         cf.AGENT_SIGHT_RADIUS:new_world.shape[1] + cf.AGENT_SIGHT_RADIUS] = new_world.copy()
            
            current_worlds[pro_id] = padded_world.copy()

        i = 1
        ########################################### tick loop ############################################
        while i < cf.EPISODE_LENGTH + 2:
            netLogoProcessManager.get_world_update()

            ################################ updating world states ###############################
            species_ids = [[] for __ in range(cf.NUM_SPECIES)]
            island_ids = [[] for __ in range(cf.NUM_ISLANDS)]
            all_new_states = [[] for __ in range(cf.NUM_SPECIES)]
            all_new_extras = [[] for __ in range(cf.NUM_SPECIES)]
            for __ in range(cf.NUM_ISLANDS - dead_islands):
                # first NUM_SPECIES + 1 values in pro_ids are used to partition ids into proper species
                pro_id, living_ids, world_updates, new_species_vals = outque.get()
                if type(living_ids) is np.ndarray:
                    living_ids = living_ids.astype(np.int16)
                    offset = living_ids[0]
                    island_ids[pro_id] = living_ids[offset:]
                    
                    if type(world_updates) is np.ndarray:
                        update_world(current_worlds[pro_id], world_updates.astype(np.int16), cf.AGENT_SIGHT_RADIUS)
                    
                    if cf.MASK_VISION:
                        new_states = get_masked_world_data(current_worlds[pro_id], new_species_vals, cf.AGENT_SIGHT_RADIUS)
                    else:
                        new_states = [current_worlds[pro_id][int(y) : int(y) + 2 * cf.AGENT_SIGHT_RADIUS + 1,
                                                             int(x) : int(x) + 2 * cf.AGENT_SIGHT_RADIUS + 1 ]
                                                                          for x, y in new_species_vals[:,0:2]]
                    
                    ###########################################################################################################################
                    # adaptation of exploration rewards mentioned in:
                    # Baker, Bowen, et al. "Emergent tool use from multi-agent autocurricula." arXiv preprint arXiv:1909.07528 (2019).
                    team2_start = living_ids[1:int(cf.NUM_SPECIES / 2)].sum()
                    apply_exploration_rewards(team1_exploration_maps[pro_id], team2_exploration_maps[pro_id], new_species_vals, team2_start)
                    ###########################################################################################################################
                    
                    # Reward for making it to the end of the episode
                    if i == cf.EPISODE_LENGTH:
                        new_species_vals[:,-1] = 1.0
                        new_species_vals[:,-2] = -1
                    
                    new_species_vals[:,0] = new_species_vals[:,0] / cf.WORLD_WIDTH
                    new_species_vals[:,1] = new_species_vals[:,1] / cf.WORLD_HEIGHT
                    
                    start = 0
                    for s in range(cf.NUM_SPECIES):
                        end = start + living_ids[s + 1]
                        id_adjustment = population_dist[:pro_id, s].sum() - population_dist[pro_id, :s].sum()
                        species_ids[s].extend(living_ids[start + offset: end + offset] + id_adjustment)
                        all_new_states[s].extend(new_states[start:end])
                        all_new_extras[s].extend(new_species_vals[start:end])
                        gen_total_rewards[pro_id, s] += new_species_vals[start:end,-1].sum() / population_dist[pro_id,s]
                        start = end
                else:
                    netLogoProcessManager.kill_island(pro_id)
                    dead_islands += 1
            
            for sx, species in enumerate(all_new_extras):
                if len(species) != 0:
                    if sx < int(cf.NUM_SPECIES / 2):
                        red_advantage[i-1] = species[0][-3]
                    else:
                        purple_advantage[i-1] = species[0][-3]
                if i >= cf.EPISODE_LENGTH: continue
                for actor in species:
                    if actor[-1] > 0:
                        action_rewards[sx,int(actor[-2])] += actor[-1]
                        actions_taken[sx,int(actor[-2])] += 1
                        if abs(actor[-1]) > max_action_rewards[sx,int(actor[-2])]:
                            max_action_rewards[sx,int(actor[-2])] = abs(actor[-1])
            
            if i == 1:
                species_manager.new_generation(species_ids, all_new_states, all_new_extras)
            else:
                species_manager.add_to_actors(species_ids, all_new_states, all_new_extras)
                
            ############################## get best actions ############################
            chosen_ids = [[] for __ in range(cf.NUM_SPECIES)]
            if cf.EXPLORATION_METHOD == "shared":
                best_actions = np.full((cf.NUM_SPECIES,cf.MAX_TOTAL_POP),-1)
                if myrng.random() > exploration_rate:
                    best_actions = species_manager.decide_actions(species_ids)
            else:
                if exploration_rate < 1.0:
                    for ix, ids in enumerate(species_ids):
                        for idnum in ids:
                            if myrng.random() > exploration_rate:
                                chosen_ids[ix].append(idnum)
                    best_actions = species_manager.decide_actions(chosen_ids)

            ############################## setting actions ##############################
            for pro_id in range(cf.NUM_ISLANDS):
                if len(island_ids[pro_id]) == 0: continue
                actions = myrng.integers(0, NUM_ACTIONS, size=len(island_ids[pro_id]))
                if exploration_rate < 1.0:
                    s = 0
                    island_adjustment = population_dist[:pro_id, s].sum()
                    species_adjustment = population_dist[pro_id, :s].sum()
                    for index, actor_id in enumerate(island_ids[pro_id]):
                        while actor_id >= population_dist[pro_id, :(s + 1)].sum():
                            s += 1
                            island_adjustment = population_dist[:pro_id, s].sum()
                            species_adjustment = population_dist[pro_id, :s].sum()
                        action_ind = actor_id + island_adjustment - species_adjustment
                        if best_actions[s][int(action_ind)] != -1:
                            best_act = best_actions[s][int(action_ind)]
                            action_frequency[s,best_act] += 1
                            actions[index] = best_act
                            
                netLogoProcessManager.send_actions(pro_id, island_ids[pro_id], actions)
                
            ############################## training ##############################
            if (training_counter % cf.TRAIN_FREQUENCY == 0 and training_counter != 0):
                species_manager.train_from_memory()
            if (training_counter >= cf.UPDATE_TARGET_FREQUENCY):
                species_manager.update_target()
                training_counter = 0
            elif (g != 0 or i >= 50):
                training_counter += 1
            ############################ tick Cleanup ############################
            i += 1
            if (sum([len(s) for s in island_ids]) < cf.MIN_ALIVE):
                break

        ############################ Process Cleanup ############################
        old_population_dist = population_dist.copy()
        population_dist = distribute_populations(cf.MAX_TOTAL_POP,
                                                 myrng,
                                                 population_dist_probs, 
                                                 gen_total_rewards.copy())

        netLogoProcessManager.wrap_up_generation(population_dist)
        
        ###################### timing and score calculations ####################
        species_stats = np.zeros((cf.NUM_SPECIES, 11))
        island_stock_stats = np.zeros((cf.NUM_ISLANDS,2))
        for n in range(cf.NUM_ISLANDS):
            # actor stats has each actors stats in the form of:
            # [ who, ticks, times-eaten, trees-harvested, stone-mined, wood-built, stone-built, bridges-built,
            # team-wood-destroyed, team-stone-destroyed, enemy-wood-destroyed, enemy-stone-destroyed]
            pro_id, actor_stats, stockpile_stats = outque.get()
            island_stock_stats[pro_id,0] = int(stockpile_stats[0])
            island_stock_stats[pro_id,1] = int(stockpile_stats[1])
            s = 0
            counter = 0
            if type(actor_stats) is np.ndarray:
                for actor in actor_stats:
                    while actor[0] > old_population_dist[pro_id,s] + counter and s + 1 < cf.NUM_SPECIES:
                        counter += old_population_dist[pro_id,s]
                        s += 1
                    species_stats[s,:] += actor[1:]
            
        species_stats[:,0] = (species_stats[:,0] / (cf.SPECIES_START_POP * cf.NUM_ISLANDS)).round(0)
        if (g % 5 == 0):
            species_manager.savemodel(str(g))
        
        for change_gen, new_exp_rate in cf.EXPLORATION_SCHEDULE:
            if g == change_gen:
                exploration_decay = new_exp_rate
        
        exploration_rate = max(exploration_rate - exploration_decay, 0.05)
        
        # if exploration_rate < 0.7:
        #     if best_gen_total_rewards.sum() * (0.8 + 0.05 * success_streak - 0.01 * fail_streak) > gen_total_rewards.sum():
        #         success_streak = 0
        #         fail_streak += 1
        #         if fail_streak >= cf.FAIL_STREAK_ALLOWANCE:
        #             fail_streak = 0
        #             species_manager.increase_learning()
        #     else:
        #         fail_streak = 0
        #         success_streak += 1
        #         if success_streak >= cf.SUCCESS_STREAK_GOAL:
        #             success_streak = 0
        #             species_manager.decrease_learning()
                
        # if gen_total_rewards.sum() > best_gen_total_rewards.sum():
        #     best_gen_total_rewards = gen_total_rewards.copy()

        gen_training_time, gen_total_training, gen_predicting_time, gen_total_predictions, average_action_sentiments, gen_data_time = species_manager.get_gen_timings()
        ################### generation summary #################
        print("##########################################################################################################################")
        print("Generation {} Summary:".format(g))
        print("red_advantage mean: ", red_advantage.mean())
        print("purple_advantage mean: ", purple_advantage.mean())
        print("exploration rate is now: ", exploration_rate)
        print("learning rate is currently: ", species_manager.get_learning_rate())
        print("total seconds for generation: ", time.time() - gen_start_time)
        print("training on {} inputs took {} seconds".format(gen_total_training, gen_training_time))
        print("predicting on {} inputs took {} seconds".format(gen_total_predictions, gen_predicting_time))
        print("time spent formatting data in speciesManager was {} seconds".format(gen_data_time))
        print("ticks survived by generation: ", i)
        print("new population dist percents:")
        for probs in (population_dist_probs * 100).round(2):
            print(probs)
        data_manager.commit_data(species_stats,
                                 action_frequency,
                                 actions_taken,
                                 action_rewards,
                                 max_action_rewards,
                                 gen_total_rewards,
                                 old_population_dist,
                                 island_stock_stats,
                                 exploration_rate,
                                 average_action_sentiments,
                                 red_advantage,
                                 purple_advantage)
        print("##########################################################################################################################")
        g += 1
        ################### end generation loop #################
    print("sending messages to terminate processes")
    data_manager.create_metadata(g, species_manager.getSpeciesModel())
    pred_vals, pred_in_states, pred_in_extras = species_manager.get_tsne_data()
    data_manager.commit_tsne_data(pred_vals, pred_in_states, pred_in_extras)
    netLogoProcessManager.kill_all()
    time.sleep(5)
    print("killing JVM...")
    mainNetLogo.kill_workspace()
    time.sleep(3)
    print("attempting to exit...")
    sys.exit("exit complete")
    exit()
    quit()
    os._exit(1)
    print("exit has catostrophically failed, why am I so bad at this?")


if __name__ == "__main__":
    main()