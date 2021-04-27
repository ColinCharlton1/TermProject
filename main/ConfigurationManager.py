# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:23:09 2021

@author: Colin
"""

# Some of these variables will heavily affect memory usage of the program
# its not very optimized to conserve memory unfortunately, actor/world memory can grow quite large
# in addition, each NetLogo Instance takes up around 500MB of RAM and there is no way I know of to decrease this
# caution is recomended if making large changes to the variables directly affecting memory
# this includes: World Size, Number of total Actors, and Max Memory

# set to true if you want to see some stats in the console as the run progresses
PRINT_STATS_TO_CONSOLE = True
SAVE_DIRECTORY = 'E:/TermProjectStorage/'

####################### Actor Control Variables #######################
NUM_SPECIES = 4 # max 10
# MAX_GENS_IN_MEM * EPISODE_LENGTH is max memory size
MAX_GENS_IN_MEM = 100
SPECIES_START_POP = 20
EXPLORATION_DECAY = 0.002 # starting value
# Each pair represents a generation number and the new exploration decay rate that will 
# be used once that generation number is reached
EXPLORATION_SCHEDULE = [] #[(20, 0.003), (40, 0.005), (60, 0.008)]

# Number of islands will be the number of processes created
# used mainly because I couldn't come up with any other way
# to run systems in parrallel on netlogo
NUM_ISLANDS = 8
# sets time to wait before starting each island
# PyNetLogo uses parrallelism for running NetLogo
# so it won't wait for the resource heavy startup to complete before passing on its lock
# this helps avoid having too many startups happening concurrently
SLEEP_TIME_BETWEEN_INITS = 3

####################### Neural Network Variables #######################
LEARNING_RATE = 0.001
BATCH_SIZE = 256
TRAIN_FREQUENCY = 3
UPDATE_TARGET_FREQUENCY = 1000
DISCOUNT_RATE = 0.7

####################### System Control Variables #######################
NUMBER_OF_GENERATIONS = 10000
EPISODE_LENGTH = 750
# ends epidodes when total living actors < MIN_ALIVE
MIN_ALIVE = 5

# Use as many as your computer can handle
# the NetLogo instances use some of their own threading, so not sure how much gain you can actually get from increasing
NUM_PROCESSES = 4

# Honestly, some values of these might crash the simulation or cause it to enter into an infinite loop
# If a value breaks it, then it just needs to be changed to something more reasonable
####################### NetLogo Simulation Variables #######################
WORLD_HEIGHT = 50
WORLD_WIDTH = 80
PATCH_SIZE = 13
STOCKPILE_NUMBER = 20 # stockpiles are the patches which are arbitrarily of the greatest value to a team
BERRY_ABUNDANCE = 200 # berries in bush = BERRY_ABUNDANCE + random (BERRY_ABUNDANCE / 2)
WOOD_CONCENTRATION = 100 # wood in tree = WOOD_CONCENTRATION + random (WOOD_CONCENTRATION / 2)
ROCK_DENSITY = 40 # rock per mine action = ROCK_DENSITY + random (ROCK_DENSITY / 2)
START_HUNGER = 1000
MAX_EAT = 100 # hunger per eat action = (MAX_EAT / 2) + random (MAX_EAT / 2)
MAX_HUNGER = 1500

# controls how close to the center of the map stone is spawned
# fraction is % of edges to avoid
# ie. 0.3 means stone avoids spawning on patches with
# x < 0.3 * max-pxcor and x > 0.7 * max-pxcor and same for y
# in range [0, 0.2]
STONE_CENTERING = 0.2

# 0: teams spawn as far away from eachother as possible
# 0.5: teams each spawn on their own side
# 1.0 teams spawn randomly across whole map
# can be any number in-between
# also affects stockpile placement to an extent:
# team stock spawn = max list 0.1 (1 - 0.1 - team-seperation)
# representing the % away from teams edge they will spawn
TEAM_SEPERATION = 0.6

# Controls how team sides are drawn
# 0: left and right split
# 1: top and bottom split
TEAM_SPLIT_STYLE = 1

######### Rewards and Bonus Tuning #########
# E: enemy  F: friendly 
# __ essentially means "dependent on"
# Actors can't destroy their own stockpiles
DEATH = -0.1
EATING = 0.0001
CUT_TREE = 0.0001
MINE = 0.0001
BUILD_WOOD = 0.1
BUILD_STONE = 0.3
BUILD_BRIDGE = 0.4
DESTROY_E_WOOD = 0.05
DESTROY_E_STONE = 0.1
DESTROY_E_STOCKPILE = 1.5
DESTROY_F = -0.1
REPEAT_TURN_PENALTY = -0.01

##### Bonus Tuning #####
# multipliers based on distance to features: bonus += val / distance
ALL_STRUCTS__CENTER_WORLD = 0.5

WOOD__WATER = 1.2
WOOD__F_STOCK = 0.3
WOOD__E_STOCK = -1

STONE__WATER = 0.25
STONE__F_STOCK = 1
STONE__E_STOCK = -0.5

DESTROY_E__E_STRUCT = 1.5
DESTROY_E__F_STRUCT = 0.5
DESTROY_E__E_STOCK = 3
DESTROY_E__F_STOCK = 1

# Meaning last struct built by that actor
LAST_STRUCT_BUILT = -1

# multipliers based on number of adjacent structures bonus += val * num_adjacent
WOOD__F_ADJ = 0.2
WOOD__E_ADJ = -0.5

STONE__F_ADJ = 0.05
STONE__E_ADJ = -0.1

BRIDGE__WATER = 0.15
BRIDGE__BRIDGE = 0.25

# hunger costs, negative is applied in NetLogo
MOVE_HCOST = 1
WATER_MOVE_HCOST = 3
CUT_WOOD_HCOST = 5
MINE_HCOST = 15
BUILD_WOOD_HCOST = 5
BUILD_STONE_HCOST = 10
BUILD_BRIDGE_HCOST = 10
DESTROY_WOOD_COST = 30
DESTROY_STONE_COST = 60
DESTROY_STOCK_COST = 10

####################### Complicated Variables #######################
# either solo or shared, solo lets each individual have a chance at a random action
# shared makes it so all actors take random actions at the same time
# solo is recommended
EXPLORATION_METHOD = "solo" 

# Warning: masking vision causes memory of actors to no longer just point to worlds stored by species manager
# memory usage gets stupidly high with large numbers of actors and a large max memory
# could potentially be fixed with a different method of storage such as memmaps from numpy
MASK_VISION = False

# must change if more than 15 islands used, each island uses the same type as its index
# so to change the world type visible in the NetLogo gui, change first entry
# adjusts the percent of the world area which will be covered by each feature
# tree and bush: 0 is 10% of green patches, each increase/decrease is a change of 1%
# rocks: 0 is 5% of all patches, each increase/decrease is a change of 0.5%
# puddles: 0 is 1% of all patches, each increase/decrease is a change of 0.5%
# river: 1 is create river, 0 is don't create river
# format [rock_adjustment, tree_adjustment, bush_adjustment, puddle_adjustment, river_toggle] 
WORLD_TYPES = [[6, 2, 2, 1, 1],
               [6, 2, 2, 1, 1],
               [8, -5, 3, 3, 1],
               [8, -5, 3, 3, 1],
               [-8, 8, 0, 2, 1],
               [2, 2, 2, 6, 0],
               [2, 2, 2, 6, 0],
               [2, 2, 2, -2, 1],
               [-8, 8, 0, 2, 1],
               [2, 2, 2, -2, 1],
               [3, 1, 1, -1, 1],
               [-5, 1, 5, 1, 1],
               [-10, 1, 1, 1, 1],
               [4, 1, 1, 5, 0],
               [-10, 3, 3, 10, 0]] 

# Experimental:
FAIL_STREAK_ALLOWANCE = 5
SUCCESS_STREAK_GOAL = 5

# LR = learning rate
FAIL_STREAK_LR_MOD = 1.1
SUCCESS_STREAK_LR_MOD = 0.98

# Dependent on other Variables, just wanted to store it here
MAX_TOTAL_POP = SPECIES_START_POP * NUM_ISLANDS

# Functions for sending config data through PyNetLogo as strings to NetLogo
def get_world_configs(pro_id):
    world_type = WORLD_TYPES[pro_id]
    world_size = [WORLD_WIDTH, WORLD_HEIGHT, PATCH_SIZE, EPISODE_LENGTH]
    world_setup = [BERRY_ABUNDANCE, WOOD_CONCENTRATION, ROCK_DENSITY, STONE_CENTERING]
    team_pop = [SPECIES_START_POP * NUM_SPECIES / 2]
    actor_setup = [STOCKPILE_NUMBER, TEAM_SEPERATION, TEAM_SPLIT_STYLE]
    actor_vars = [START_HUNGER, MAX_EAT, MAX_HUNGER]
    
    return " " + " ".join(map(str,(world_type + world_size + world_setup + team_pop + actor_setup + actor_vars)))

def get_reward_configs():
    rewards1 = [DEATH, EATING, CUT_TREE, MINE, BUILD_WOOD, BUILD_STONE, BUILD_BRIDGE]
    rewards2 = [DESTROY_E_WOOD, DESTROY_E_STONE, DESTROY_E_STOCKPILE, DESTROY_F]
    wood_b = [WOOD__WATER, WOOD__F_STOCK, WOOD__E_STOCK]
    stone_b = [STONE__WATER, STONE__F_STOCK, STONE__E_STOCK]
    misc_b = [ALL_STRUCTS__CENTER_WORLD, LAST_STRUCT_BUILT]
    dest_b1 = [DESTROY_E__E_STRUCT, DESTROY_E__F_STRUCT]
    dest_b2 = [DESTROY_E__E_STOCK, DESTROY_E__F_STOCK]
    adj_b1 = [WOOD__F_ADJ, WOOD__E_ADJ, STONE__F_ADJ, STONE__E_ADJ]
    adj_b2 = [BRIDGE__WATER, BRIDGE__BRIDGE]
    penalties = [REPEAT_TURN_PENALTY]
    costs1 = [MOVE_HCOST, WATER_MOVE_HCOST, CUT_WOOD_HCOST, MINE_HCOST, BUILD_WOOD_HCOST]
    costs2 = [BUILD_STONE_HCOST, BUILD_BRIDGE_HCOST, DESTROY_WOOD_COST, DESTROY_STONE_COST, DESTROY_STOCK_COST]
    
    return " " + " ".join(map(str,(rewards1 + rewards2 + wood_b + stone_b + misc_b + dest_b1 +
                                   dest_b2 + adj_b1 + adj_b2 + penalties + costs1 + costs2)))

# These changes are very related
# if changed, make sure to compile an individual species model and use model.summary() 
# to check if it works and if the kernels line-up with the vision size
# not particularly recommended to change
AGENT_SIGHT_RADIUS = 15
TIMESTEPS_IN_OBSERVATIONS = 4

# some inspiration for architecture taken from:
# Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision."
# Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
# - https://arxiv.org/pdf/1512.00567.pdf
# 0 filters creates MaxPool layer
NETWORK_CONV_ARCHITECTURE = [[16, (1,4,1)],[16, (1,1,4)],[24, (1,3,1)],[24, (1,1,3)], [24, (2,1,1)],
                             [0, (1,2,2)],
                             [48, (1,3,1)],[48, (1,1,3)],[64, (1,2,1)],[64, (1,1,2)],[64, (2,1,1)],
                             [0, (1,2,2)],
                             [96, (1,1,3)],[96, (1,3,1)],[128, (1,1,3)],[128, (1,3,1)],[128, (2,1,1)]]

EXTRA_CONV_LAYERS = [(32,2), (48,2), (64,2)]

DROPOUT_RATE = 0.2
DENSE_LAYERS = [192, 192]


# section for testing network architecture and speed
if __name__ == "__main__":
    # import os
    # import logging
    # logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from ConvFeatureModel import createFullSpeciesModel, createFlexibleConvFeatureModel, createExperimentalFullSpeciesModel
    import tensorflow as tf
    import time
    import numpy as np
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    NUM_ACTIONS = 13 # must be 13
    WORLD_FEATURES = 3 # must be 3
    num_extras = 10
    dims = AGENT_SIGHT_RADIUS * 2 + 1
    
    # testmodel = createFlexibleConvFeatureModel(TIMESTEPS_IN_OBSERVATIONS, dims, WORLD_FEATURES, num_extras - 2, NUM_ACTIONS, "test")
    # testmodel.summary()
    
    species_list = ["red1", "red2", "purple1", "purple2"]
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    testmodel = createFullSpeciesModel(NUM_SPECIES, TIMESTEPS_IN_OBSERVATIONS, dims, WORLD_FEATURES, num_extras - 2, NUM_ACTIONS, "fullModel", species_list)
    testmodel.compile(tf.optimizers.Adam(learning_rate=0.001), "mse")
    # testmodel.summary()
    # testmodel = createExperimentalFullSpeciesModel(NUM_SPECIES, TIMESTEPS_IN_OBSERVATIONS, dims, WORLD_FEATURES, num_extras - 2, NUM_ACTIONS, "fullModel", species_list)
    # testmodel.compile(tf.optimizers.RMSprop(learning_rate=0.001), "mse")
    
    test_ticks = 750
    train_frequency = 5
    # test_batch_sizes = [25, 50, 75, 100, 128, 150, 175, 200, 250, 300, 400, 500]
    test_batch_sizes = [1, 8, 16, 32, 64, 96, 120, 128, 256]
    # test_batch_sizes = [128]
    rng = np.random.default_rng()
    states = np.ndarray(shape = (test_batch_sizes[-1], TIMESTEPS_IN_OBSERVATIONS, dims, dims, 3),
                        buffer=rng.random(size=test_batch_sizes[-1]*TIMESTEPS_IN_OBSERVATIONS*dims*dims*3, dtype=np.float32),
                        dtype=np.float32)
    
    extras = np.ndarray(shape = (test_batch_sizes[-1], TIMESTEPS_IN_OBSERVATIONS, num_extras - 2),
                        buffer=rng.random(size=test_batch_sizes[-1]*TIMESTEPS_IN_OBSERVATIONS*(num_extras - 2), dtype=np.float32),
                        dtype=np.float32)

    for batch in test_batch_sizes:
        test_states = states[:batch].copy()
        test_extras = extras[:batch].copy()
        preds = testmodel([[test_states, test_extras],[test_states, test_extras],[test_states, test_extras],[test_states, test_extras]], training=False)
        targets = [p + 0.5 for p in preds]
        testmodel.train_on_batch([[test_states, test_extras],[test_states, test_extras],[test_states, test_extras],[test_states, test_extras]], targets)
        predict_time = 0
        train_time = 0
        total_trains = 0
        for i in range(1, test_ticks + 1):
            start_time = time.time()
            preds = testmodel([[test_states, test_extras],[test_states, test_extras],[test_states, test_extras],[test_states, test_extras]], training=False)
            predict_time += time.time() - start_time
            if i % train_frequency == 0:
                targets = [p + 0.5 for p in preds]
                start_time = time.time()
                testmodel.train_on_batch([[test_states, test_extras],[test_states, test_extras],[test_states, test_extras],[test_states, test_extras]], targets)
                train_time += time.time() - start_time
                total_trains += 1
            test_states *= 0.999999
            test_extras *= 0.999999
        print("batch size {} total predict_time for {} test ticks: {}".format(batch, test_ticks, predict_time))
        print("batch size {} total train_time for {} training sessions: {}".format(batch, total_trains, train_time))