# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:15:39 2021

@author: Colin
"""

import ConfigurationManager as cf
import numpy as np
import os

class DataManager():
    def __init__(self, species_list, run_name, verbose=False):
        self.run_name = run_name
        self.verbose = verbose
        self.dataDirName = 'rawdata'
        self.counter = 0
        self.species_list = species_list
        # avg_species_history stores a list of variables for each species
        # the data is averaged over 5 gens, each set of 5 is completely seperate
        # data is in form: [ reward, wood_built, stone_built, ewood_destroyed, estone_destroyed, fwood_destroyed, fstone_destroyed]
        self.avg_species_history = [[] for __ in range(cf.NUM_SPECIES)]
        
        prediction_strings = ["move_up","move_left","move_right","move_down","turn_right","turn_left","eat","cut","mine","build_wood","build_stone","build_bridge","destroy"]
        datanames = ["avg_ticks", "times_eaten", "trees_cut", "stone_mined", "wood_built", "stone_built", "bridges_built", "twood_destroyed", "tstone_destroyed", "ewood_destroyed", "estone_destroyed"]
        self.condensed_prediction_strings = ["mov_up", "mov_le", "mov_ri", "mov_do", "trn_ri", "trn_le","_eat__","_cut__","_mine_","bld_wd","bld_st","bld_br","destry"]
    
        self.prediction_headers = ', '.join([s + "_" + a for s in species_list for a in prediction_strings])
        self.species_data_headers = ', '.join([s + "_" + a for s in species_list for a in datanames])
        self.reward_headers = ', '.join(species_list * cf.NUM_ISLANDS)
        self.stockpile_headers = ', '.join([s + "_" + str(a) for a in range(cf.NUM_ISLANDS) for s in ["team1_island", "team2_island"]])
        self.pop_dist_headers = ', '.join([s + "_island" + str(island) for island in range(cf.NUM_ISLANDS) for s in species_list])
        
        self.species_data_filename = self.dataDirName + "/species_data_" + self.run_name + ".csv"
        self.prediction_data_filename = self.dataDirName + "/prediction_data_" + self.run_name + ".csv"
        self.reward_data_filename = self.dataDirName + "/reward_data_" + self.run_name + ".csv"
        self.pop_dist_filename = self.dataDirName + "/population_data_" + self.run_name + ".csv"
        self.stockpile_filename = self.dataDirName + "/stockpile_data_" + self.run_name + ".csv"
        
    
    
    def __update_averages(self, species_stats, reward_stats):
        if self.counter % 5 == 0:
            if (self.verbose and self.counter != 0): 
                self.__print_averages()
            for s in range(cf.NUM_SPECIES):
                new_avgs = np.zeros(7)
                new_avgs[0] += reward_stats[:,s].sum()
                new_avgs[1:7] += species_stats[s,5:11]
                self.avg_species_history[s].append(new_avgs)
        else:
            for s in range(cf.NUM_SPECIES):
                self.avg_species_history[s][-1][0] += reward_stats[:,s].sum()
                self.avg_species_history[s][-1][1:7] += species_stats[s,5:11]
            

    def commit_data(self, species_stats, prediction_frequencies, reward_stats, population_dist, stockpile_stats):
        
        if self.verbose:
            self.__print_islands(reward_stats, population_dist, stockpile_stats)
            self.__print_predictions(prediction_frequencies)
            
        self.__update_averages(species_stats, reward_stats)
        
        if self.counter == 0:
            self.counter = 1
            self.__initialize_savedata(species_stats, prediction_frequencies, reward_stats, population_dist, stockpile_stats)
            
        else:
            self.counter += 1
            self.__update_data(species_stats, prediction_frequencies, reward_stats, population_dist, stockpile_stats)
    
    def __initialize_savedata(self, species_stats, prediction_frequencies, reward_stats, population_dist, stockpile_stats):
        
        if not os.path.exists(self.dataDirName):
            os.mkdir(self.dataDirName)
            
        with open(self.species_data_filename, 'wb') as f:
            np.savetxt(f, species_stats.reshape(1,-1), header=self.species_data_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.prediction_data_filename, 'wb') as f:
            np.savetxt(f, prediction_frequencies.reshape(1,-1), header=self.prediction_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.reward_data_filename, 'wb') as f:
            np.savetxt(f, reward_stats.reshape(1,-1), header=self.reward_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.pop_dist_filename, 'wb') as f:
            np.savetxt(f, population_dist.reshape(1,-1), header=self.pop_dist_headers, delimiter=",", newline="\n")
            f.flush()
            
        with open(self.stockpile_filename, 'wb') as f:
            np.savetxt(f, stockpile_stats, header=self.stockpile_headers, delimiter=",", newline="\n")
            f.flush()
    
    def __update_data(self, species_stats, prediction_frequencies, reward_stats, population_dist, stockpile_stats):
        with open(self.species_data_filename, 'ab') as f:
            np.savetxt(f, species_stats.reshape(1,-1), delimiter=",", newline="\n")
            f.flush()
            
        with open(self.prediction_data_filename, 'ab') as f:
            np.savetxt(f, prediction_frequencies.reshape(1,-1), delimiter=",", newline="\n")
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

    def __print_averages(self):
        print("new 5 generation species averages: ")
        for ix, species in enumerate(self.avg_species_history):
            print("species {}:".format(self.species_list[ix]))
            maxlen = max(len(str(np.asarray(species).round(0).max())), 5)
            for set_num, gen5 in enumerate(species):
                print("gen {:>3}->{:<3}:".format(5 * set_num, 5 * (set_num + 1) - 1), end="| ")
                for val in gen5:
                    print("{:>{}}".format(int(val), maxlen), end=" ")   
                print("|")
                
    def __print_islands(self, reward_stats, population_dist, stockpile_stats):
        print("island data: ")
        for i in range(cf.NUM_ISLANDS):
            print("island {:>2} with stockpile score ({:>3}:{:<3})".format(i, int(stockpile_stats[i,0]), int(stockpile_stats[i,1])))
            print("population dist: ", end="| ")
            for s in range(cf.NUM_SPECIES):
                print("{:^8}".format(population_dist[i,s]), end=" ")
            print("|")
            print("        rewards: ", end="| ")
            for s in range(cf.NUM_SPECIES):
                print("{:^8}".format(reward_stats[i,s].round(2)), end=" ")
            print("|")
            
    def __print_predictions(self, prediction_frequencies):
        print("prediction action frequencies: ")
        print("           ", end="| ")
        maxlen = max(len(str(prediction_frequencies.max())), 6)
        for action in self.condensed_prediction_strings:
            print("{:>{}}".format(action, maxlen), end=" ")
        print("|")
        for s in range(cf.NUM_SPECIES):
            print("species {:>2}:".format(s), end="| ")
            for pred in prediction_frequencies[s]:
                print("{:>{}}".format(pred, maxlen), end=" ")
            print("|")