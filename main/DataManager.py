# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:15:39 2021

@author: Colin
"""

import ConfigurationManager as cf
import numpy as np
import os

class DataManager():
    def __init__(self, species_list, runDirName):
        # set run name so different runs are automatically recorded seperately
        self.runDirName = runDirName
        self.verbose = cf.PRINT_STATS_TO_CONSOLE
        self.dataDirName = runDirName + '/rawdata'
        if not os.path.exists(self.dataDirName):
            os.mkdir(self.dataDirName)
        self.counter = 0
        self.species_headers = species_list[:int(cf.NUM_SPECIES / 2)]
        for s in range(int(cf.NUM_SPECIES / 2)):
            self.species_headers.append("purp" + str(s + 1))
        # avg_species_history stores a list of variables for each species
        # the data is averaged over 5 gens, each set of 5 is completely seperate
        # data is in form: [ reward, wood_built, stone_built, bridges_built, team_wood_destroyed, team_stone_destroyed, enemy_wood_destroyed, enemy_stone_destroyed]
        self.avg_species_history = [[] for __ in range(cf.NUM_SPECIES)]
        
        prediction_strings = ["move_up","move_left","move_right","move_down","turn_right","turn_left","eat","cut","mine","build_wood","build_stone","build_bridge","destroy"]
        datanames = ["avg_ticks", "times_eaten", "trees_cut", "stone_mined", "wood_built", "stone_built", "bridges_built", "twood_destroyed", "tstone_destroyed", "ewood_destroyed", "estone_destroyed"]
        self.condensed_prediction_strings = ["mve_up", "mve_lt", "mve_rt", "mve_dn", "trn_rt", "trn_lt","eat","cut","mine","bld_wd","bld_st","bld_br","dstroy"]
        self.species_stat_labels = [ "reward", "wood_built", "stne_built", "brdg_built", "tm_wd_dest", "tm_st_dest", "en_wd_dest", "enmys_dest"]
        
        self.prediction_headers = ', '.join([s + "_" + a for s in species_list for a in prediction_strings])
        self.species_data_headers = ', '.join([s + "_" + a for s in species_list for a in datanames])
        self.reward_headers = ', '.join(species_list * cf.NUM_ISLANDS)
        self.stockpile_headers = ', '.join([s + "_" + str(a) for a in range(cf.NUM_ISLANDS) for s in ["team1_island", "team2_island"]])
        self.pop_dist_headers = ', '.join([s + "_island" + str(island) for island in range(cf.NUM_ISLANDS) for s in species_list])
        
        self.species_data_filename = self.dataDirName + "/species_data.csv"
        self.prediction_data_filename = self.dataDirName + "/prediction_data.csv"
        self.action_reward_data_filename = self.dataDirName + "/action_reward_data.csv"
        self.reward_data_filename = self.dataDirName + "/reward_data.csv"
        self.pop_dist_filename = self.dataDirName + "/population_data.csv"
        self.stockpile_filename = self.dataDirName + "/stockpile_data.csv"
        self.exploration_filename = self.dataDirName + "/exploration_history.csv"
        self.red_advantage_filename = self.dataDirName + "/red_advantage_data.csv"
        self.purple_advantage_filename = self.dataDirName + "/purple_advantage_history.csv"
    
    def __update_averages(self, species_stats, reward_stats):
        if self.counter % 5 == 0:
            if (self.verbose and self.counter != 0):
                self.__print_averages()
            for s in range(cf.NUM_SPECIES):
                new_avgs = np.zeros(8)
                new_avgs[0] += reward_stats[:,s].mean() / 5
                new_avgs[1:8] += species_stats[s,4:11] / 5
                self.avg_species_history[s].append(new_avgs)
            if self.counter == 0: self.counter = 1
            
        else:
            for s in range(cf.NUM_SPECIES):
                self.avg_species_history[s][-1][0] += reward_stats[:,s].mean() / 5
                self.avg_species_history[s][-1][1:8] += species_stats[s,4:11] / 5
            

    def commit_data(self, species_stats,   prediction_frequencies,   actions_taken,   action_rewards,    max_action_rewards,
                          reward_stats,   population_dist,   stockpile_stats,    exploration_rate,   average_action_sentiments, red_advantage, purple_advantage):
        
        if self.counter == 0:
            self.__initialize_savedata(species_stats, prediction_frequencies, actions_taken, action_rewards, reward_stats, population_dist, stockpile_stats, exploration_rate, red_advantage, purple_advantage)
            
        else:
            self.counter += 1
            self.__update_data(species_stats, prediction_frequencies, actions_taken, action_rewards, reward_stats, population_dist, stockpile_stats, exploration_rate, red_advantage, purple_advantage)
            
        if self.verbose:
            self.__print_islands(reward_stats, population_dist, stockpile_stats)
            self.__print_predictions(prediction_frequencies, actions_taken, action_rewards, average_action_sentiments, max_action_rewards)
            
        self.__update_averages(species_stats, reward_stats)
        
        
    
    def __initialize_savedata(self, species_stats, prediction_frequencies, actions_taken, action_rewards, reward_stats, population_dist, stockpile_stats, exploration_rate, red_advantage, purple_advantage):
        
        if not os.path.exists(self.dataDirName):
            os.mkdir(self.dataDirName)
            
        with open(self.species_data_filename, 'wb') as f:
            np.savetxt(f, species_stats.reshape(1,-1), header=self.species_data_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.prediction_data_filename, 'wb') as f:
            np.savetxt(f, prediction_frequencies.reshape(1,-1), header=self.prediction_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.action_reward_data_filename, 'wb') as f:
            np.savetxt(f, action_rewards.reshape(1,-1), header=self.prediction_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.reward_data_filename, 'wb') as f:
            np.savetxt(f, reward_stats.reshape(1,-1), header=self.reward_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.pop_dist_filename, 'wb') as f:
            np.savetxt(f, population_dist.reshape(1,-1), header=self.pop_dist_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.stockpile_filename, 'wb') as f:
            np.savetxt(f, stockpile_stats.reshape(1,-1), header=self.stockpile_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.exploration_filename, 'wb') as f:
            np.savetxt(f, np.asarray([exploration_rate]).reshape(1,-1), header="exploration_rate", delimiter=",", newline="\n")
            f.flush()
            
        with open(self.red_advantage_filename, 'wb') as f:
            np.savetxt(f, red_advantage.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
            
        with open(self.purple_advantage_filename, 'wb') as f:
            np.savetxt(f, purple_advantage.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
    
    def __update_data(self, species_stats, prediction_frequencies, actions_taken, action_rewards, reward_stats, population_dist, stockpile_stats, exploration_rate, red_advantage, purple_advantage):
        with open(self.species_data_filename, 'ab') as f:
            np.savetxt(f, species_stats.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
            
        with open(self.prediction_data_filename, 'ab') as f:
            np.savetxt(f, prediction_frequencies.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
            
        with open(self.action_reward_data_filename, 'ab') as f:
            np.savetxt(f, action_rewards.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
            
        with open(self.reward_data_filename, 'ab') as f:
            np.savetxt(f, reward_stats.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
            
        with open(self.pop_dist_filename, 'ab') as f:
            np.savetxt(f, population_dist.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
          
        with open(self.stockpile_filename, 'ab') as f:
            np.savetxt(f, stockpile_stats.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
            
        with open(self.exploration_filename, 'ab') as f:
            np.savetxt(f, np.asarray([exploration_rate]).reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
            
        with open(self.red_advantage_filename, 'ab') as f:
            np.savetxt(f, red_advantage.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
            
        with open(self.purple_advantage_filename, 'ab') as f:
            np.savetxt(f, purple_advantage.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()

    def __print_averages(self):
        title = " 5 generation species averages "
        sub_title = "species {:5.5}:"
        row_label = "gen{:>4}->{:<4}:"
        ind_width = max(len(sub_title.format("0123456789")), len(row_label.format(0,0)))
        max_head_len = max([len(h) for h in self.species_stat_labels])
        max_rew_len = len(str(np.asarray(self.avg_species_history)[:,:,0].round(2).max())) + 2
        max_int_len = len(str(int(np.asarray(self.avg_species_history).max()))) + 2
        col_width = max(max_head_len, max_rew_len, max_int_len)
        sides_size = (ind_width + 1) + (col_width + 1) * len(self.species_stat_labels) - len(title)
        self.__print_title(title, sides_size)
        for ix, species in enumerate(self.avg_species_history):
            values = []
            row_labels = []
            for set_num, gen5 in enumerate(species):
                row_labels.append(row_label.format(5 * set_num, 5 * (set_num + 1) - 1))
                values.append([gen5[0].round(2)] + gen5[1:].astype(np.int32).tolist())
            sub = sub_title.format(self.species_headers[ix])
            self.__print_sub_grid(sub, self.species_stat_labels, row_labels, values, ind_width, col_width)
            print("-" * (sides_size + len(title)))
                
    def __print_islands(self, reward_stats, population_dist, stockpile_stats):
        title = " island data "
        title2 = " island {:<2} "
        sub_title = "stocks({:>3}:{:<3}):"
        row_labels = ["population dist:", "average rewards:"]
        ind_width = max(len(sub_title.format(123, 123)), max([len(l) for l in row_labels]))
        max_head_len = max([len(h) for h in self.species_headers])
        max_rew_len = max([len(str(r)) for r in reward_stats.flatten().round(2)]) + 2
        max_pop_len = len(str(int(population_dist.max()))) + 2
        col_width = max(max_head_len, max_rew_len, max_pop_len)
        sides_size = (ind_width + 1) + (col_width + 1) * (len(self.species_headers)) - len(title)
        self.__print_title(title, sides_size)
        for i in range(cf.NUM_ISLANDS):
            sub_side_size = (ind_width + 1) + (col_width + 1) * len(self.species_headers) - len(title2.format(i))
            self.__print_title(title2.format(i), sub_side_size)
            rows = [population_dist[i,:],reward_stats[i,:].round(2)]
            sub = sub_title.format(int(stockpile_stats[i,0]), int(stockpile_stats[i,1]))
            self.__print_sub_grid(sub, self.species_headers, row_labels, rows, ind_width, col_width)
    
            
    def __print_predictions(self, prediction_frequencies, actions_taken, action_rewards, average_action_sentiments, max_action_rewards):
        title = " prediction action frequencies "
        sub_title = "species"
        row_labels = [spc + ":" for spc in self.species_headers]
        ind_width = max(len(sub_title), max([len(l) for l in row_labels]))
        max_head_len = max([len(h) for h in self.condensed_prediction_strings])
        max_pred_len = len(str(int(prediction_frequencies.max()))) + 2
        col_width = max(max_head_len, max_pred_len)
        sides_size = (ind_width + 1) + (col_width + 1) * len(self.condensed_prediction_strings) - len(title)
        self.__print_title(title, sides_size)
        self.__print_sub_grid(sub_title, self.condensed_prediction_strings, row_labels,
                              prediction_frequencies, ind_width, col_width)
        
        title = " average positive rewards for actions "
        val_format = "{:0{}.4f}"
        rows = np.divide(action_rewards, actions_taken, out=np.zeros_like(action_rewards), where=actions_taken!=0).round(4)
        max_reward_length = max([len(val_format.format(rew, len(str(rew)))) for rew in rows.flatten()])
        col_width = max(max_head_len, max_reward_length)
        sides_size = (ind_width + 1) + (col_width + 1) * len(self.condensed_prediction_strings) - len(title)
        self.__print_title(title, sides_size)
        self.__print_condensed_sub_grid(sub_title, self.condensed_prediction_strings, row_labels,
                                        rows, ind_width, col_width)
        
        title = " max reward for each action "
        val_format = "{:0{}.4f}"
        rows = max_action_rewards.round(4)
        max_reward_length = max([len(val_format.format(rew, len(str(rew)))) for rew in rows.flatten()])
        col_width = max(max_head_len, max_reward_length)
        sides_size = (ind_width + 1) + (col_width + 1) * len(self.condensed_prediction_strings) - len(title)
        self.__print_title(title, sides_size)
        self.__print_condensed_sub_grid(sub_title, self.condensed_prediction_strings, row_labels,
                                        rows, ind_width, col_width)
        
        title = " average predicted value for each action "
        max_value_length = max([len(val_format.format(val, len(str(val)))) for val in average_action_sentiments.flatten().round(4)])
        col_width = max(max_head_len, max_value_length)
        sides_size = (ind_width + 1) + (col_width + 1) * len(self.condensed_prediction_strings) - len(title)
        self.__print_title(title, sides_size)
        self.__print_condensed_sub_grid(sub_title, self.condensed_prediction_strings, row_labels,
                                        average_action_sentiments.round(4), ind_width, col_width)
    
    def __print_title(self, title, sides_size):   
        print("-" * int(sides_size / 2) + title + "-" * int(sides_size / 2))
    
    def __print_sub_grid(self, sub_title, headers, row_labels, rows, index_width, column_width):
        print("{:<{}}".format(sub_title, index_width), end = "|") 
        for header in headers:
            print("{:^{}}".format(header, column_width), end="|") 
        print("")
        for i, vals in enumerate(rows):
            print("{:>{}}".format(row_labels[i], index_width), end="|")
            for val in vals:
                print(" {:>{}} ".format(val, column_width-2), end="|")   
            print("")
            
    def __print_condensed_sub_grid(self, sub_title, headers, row_labels, rows, index_width, column_width):
        print("{:<{}}".format(sub_title, index_width), end = "|") 
        for header in headers:
            print("{:^{}}".format(header, column_width), end="|") 
        print("")
        for i, vals in enumerate(rows):
            print("{:>{}}".format(row_labels[i], index_width), end="|")
            for val in vals:
                print("{:0{}.4f}".format(val, column_width), end="|")   
            print("")
     
    def commit_tsne_data(self, pred_vals, pred_in_states, pred_in_extras):
        np.save(self.runDirName + "/tsne_pred_vals.npy", np.asarray(pred_vals))
        np.save(self.runDirName + "/tsne_pred_states.npy", np.asarray(pred_in_states))
        np.save(self.runDirName + "/tsne_pred_extras.npy", np.asarray(pred_in_extras))
            
    def create_metadata(self, generations, model):
        meta_file = self.runDirName + "/metadata.txt"
        with open(meta_file, 'w') as f:
            f.write("Total Generations Ran For: " + str(generations) + "\n")
            
            f.write("_____________World_Info_____________\n")
            f.write("Number of Islands: " + str(cf.NUM_ISLANDS) + "\n")
            f.write("Number of Species: " + str(cf.NUM_SPECIES) + "\n")
            f.write("Species Start Population: " + str(cf.SPECIES_START_POP) + "\n")
            f.write("Actor Start Hunger: " + str(cf.START_HUNGER) + "\n")
            f.write("Actor Max Eating: " + str(cf.MAX_EAT) + "\n")
            f.write("Actor Max Hunger: " + str(cf.MAX_HUNGER) + "\n")
            f.write("Masking Vision: " + str(cf.MASK_VISION) + "\n")
            f.write("Team Seperation: " + str(cf.TEAM_SEPERATION) + "\n")
            f.write("Team Split Style: " + str(cf.TEAM_SPLIT_STYLE) + "\n")
            f.write("Episode Length: " + str(cf.EPISODE_LENGTH) + "\n")
            f.write("World Size (Width x Height): " + str(cf.WORLD_WIDTH) + " x " + str(cf.WORLD_HEIGHT) + "\n")
            f.write("Number of Stockpiles: " + str(cf.STOCKPILE_NUMBER) + "\n")
            f.write("Berry Abundance: " + str(cf.BERRY_ABUNDANCE) + "\n")
            f.write("Wood Concentration: " + str(cf.WOOD_CONCENTRATION) + "\n")
            f.write("Rock Density: " + str(cf.ROCK_DENSITY) + "\n")
            f.write("Stone Centering: " + str(cf.STONE_CENTERING) + "\n")
            f.write("World Types:")
            for wt in cf.WORLD_TYPES:
                f.write(str(wt) + "\n")
            
            f.write("____________Network_Info____________\n")
            f.write("Max Memory Size: " + str(cf.MAX_MEM_SIZE) + "\n")
            f.write("Exploration Decay Rate: " + str(cf.EXPLORATION_DECAY) + "\n")
            f.write("Exploration Method: " + str(cf.EXPLORATION_METHOD) + "\n")
            f.write("Discount Rate: " + str(cf.DISCOUNT_RATE) + "\n")
            f.write("Learning Rate: " + str(cf.LEARNING_RATE) + "\n")
            f.write("Training Batch Size: " + str(cf.BATCH_SIZE) + "\n")
            f.write("Training Frequency: " + str(cf.TRAIN_FREQUENCY) + "\n")
            f.write("Target Update Frequency: " + str(cf.UPDATE_TARGET_FREQUENCY) + "\n")
            f.write("Agent Sight Radius: " + str(cf.AGENT_SIGHT_RADIUS) + "\n")
            f.write("Timesteps in Observations: " + str(cf.TIMESTEPS_IN_OBSERVATIONS) + "\n")
            f.write("Network Architecture:")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            
            f.write("____________Rewards_Info____________\n")
            f.write("Dying: " + str(cf.DEATH) + "\n")
            f.write("Eating Berries: " + str(cf.EATING) + "\n")
            f.write("Cutting Tree: " + str(cf.CUT_TREE) + "\n")
            f.write("Mining Stone: " + str(cf.MINE) + "\n")
            f.write("Building Wood Structure: " + str(cf.BUILD_WOOD) + "\n")
            f.write("Building Stone Structure: " + str(cf.BUILD_STONE) + "\n")
            f.write("Building Bridge: " + str(cf.BUILD_BRIDGE) + "\n")
            f.write("Destoying Enemy Wood Structure: " + str(cf.DESTROY_E_WOOD) + "\n")
            f.write("Destoying Enemy Stone Structure: " + str(cf.DESTROY_E_STONE) + "\n")
            f.write("Destoying Enemy Stockpile: " + str(cf.DESTROY_E_STOCKPILE) + "\n")
            f.write("Destoying Friendly Structure: " + str(cf.DESTROY_F) + "\n")
            
            f.write("____________Bonuses_Info____________\n")
            f.write("Distance to Center of World: " + str(cf.ALL_STRUCTS__CENTER_WORLD) + "\n")
            f.write("Distance to Last Structure: " + str(cf.LAST_STRUCT_BUILT) + "\n")
            f.write("Wood Distance to Water Patch: " + str(cf.WOOD__WATER) + "\n")
            f.write("Wood Distance to Friendly Stockpile: " + str(cf.WOOD__F_STOCK) + "\n")
            f.write("Wood Distance to Enemy Stockpile: " + str(cf.WOOD__E_STOCK) + "\n")
            f.write("Wood Number of Adjacent Friendly Structures: " + str(cf.WOOD__F_ADJ) + "\n")
            f.write("Wood Number of Adjacent Enemy Structures: " + str(cf.WOOD__E_ADJ) + "\n")
            f.write("Stone Distance to Water Patch: " + str(cf.STONE__WATER) + "\n")
            f.write("Stone Distance to Friendly Stockpile: " + str(cf.STONE__F_STOCK) + "\n")
            f.write("Stone Distance to Enemy Stockpile: " + str(cf.STONE__E_STOCK) + "\n")
            f.write("Stone Number of Adjacent Friendly Structures: " + str(cf.STONE__F_ADJ) + "\n")
            f.write("Stone Number of Adjacent Enemy Structures: " + str(cf.STONE__E_ADJ) + "\n")
            f.write("Bridge Number of Adjacent Water Patches: " + str(cf.BRIDGE__WATER) + "\n")
            f.write("Bridge Number of Adjacent Bridges: " + str(cf.BRIDGE__BRIDGE) + "\n")
            
            f.write("__________Hunger_Cost_Info__________\n")
            f.write("Move: " + str(cf.MOVE_HCOST) + "\n")
            f.write("Move in Water: " + str(cf.WATER_MOVE_HCOST) + "\n")
            f.write("Cut Wood: " + str(cf.CUT_WOOD_HCOST) + "\n")
            f.write("Mine Stone: " + str(cf.MINE_HCOST) + "\n")
            f.write("Build Wood Structure: " + str(cf.BUILD_WOOD_HCOST) + "\n")
            f.write("Build Stone Structure: " + str(cf.BUILD_STONE_HCOST) + "\n")
            f.write("Build Bridge: " + str(cf.BUILD_BRIDGE_HCOST) + "\n")
            f.write("Destroy Wood Structure: " + str(cf.DESTROY_WOOD_COST) + "\n")
            f.write("Destroy Stone Structure: " + str(cf.DESTROY_STONE_COST) + "\n")
            f.write("Destroy Stockpile: " + str(cf.DESTROY_STOCK_COST) + "\n")
            f.flush()
        