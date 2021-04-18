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

####################### Actor Control Variables #######################
NUM_SPECIES = 4 # max 10
SPECIES_START_POP = 30
EXPLORATION_DECAY = 0.001
MAX_MEM_SIZE = 40000
# Number of islands will be the number of processes created
# used mainly because I couldn't come up with any other way
# to run systems in parrallel on netlogo
NUM_ISLANDS = 5

####################### Neural Network Variables #######################
LEARNING_RATE = 0.0008
BATCH_SIZE = 512
TRAIN_FREQUENCY = 5
UPDATE_TARGET_FREQUENCY = 200
DISCOUNT_RATE = 0.8


####################### System Control Variables #######################
NUMBER_OF_GENERATIONS = 10000
EPISODE_LENGTH = 600

# Use as many as your computer can handle
# the NetLogo instances use some of their own threading, so not sure how much gain you can actually get from increasing
NUM_PROCESSES = 6 

####################### NetLogo Simulation Variables #######################
WORLD_HEIGHT = 60
WORLD_WIDTH = 90
PATCH_SIZE = 12
STOCKPILE_NUMBER = 15 # stockpiles are the patches which are arbitrarily of the greatest value to a team
BERRY_ABUNDANCE = 200 # berries in bush = BERRY_ABUNDANCE + random (BERRY_ABUNDANCE / 2)
WOOD_CONCENTRATION = 100 # wood in tree = WOOD_CONCENTRATION + random (WOOD_CONCENTRATION / 2)
ROCK_DENSITY = 40 # rock per mine action = ROCK_DENSITY + random (ROCK_DENSITY / 2)
START_HUNGER = 750
MAX_EAT = 100 # hunger per eat action = (MAX_EAT / 2) + random (MAX_EAT / 2)
MAX_HUNGER = 1500

# controls how close to the center of the map stone is spawned
# negative values force stone towards edges
# fraction is % of edges to avoid
# ie. 0.3 means stone avoids spawning on patches with
# x < 0.3 * max-pxcor and x > 0.7 * max-pxcor and same for y
# in range [-0.4, 0.4]
STONE_CENTERING = 0.2

# 0: teams spawn as far away from eachother as possible
# 0.5: teams each spawn on their own side
# 1.0 teams spawn randomly across whole map
# can be any number in-between
# also affects stockpile placement to an extent:
# team stock spawn = max list 0.1 (1 - 0.1 - team-seperation)
# representing the % away from teams edge they will spawn
TEAM_SEPERATION = 0.75

# Controls how team sides are drawn
# 0: left and right split
# 1: top and bottom split
# 2: diagonal split
TEAM_SPLIT_STYLE = 0

######### Rewards and Bonus Tuning #########
# E: enemy  F: friendly
# __ essentially means "dependent on"
# Actors can't destroy their own stockpiles
EATING = 0
CUT_TREE = 0
MINE = 0
BUILD_WOOD = 0.05
BUILD_STONE = 0.1
BUILD_BRIDGE = 0.2
DESTROY_E_WOOD = 0.1
DESTROY_E_STONE = 0.3
DESTROY_E_STOCKPILE = 5
DESTROY_F = -0.03
REPEAT_TURN_PENALTY = -0.03

# Bonus Tuning

# multipliers based on distance to features: bonus += val / distance
WOOD__WATER = 2
WOOD__F_STOCK = 3
WOOD__E_STOCK = -3
STONE__WATER = 2
STONE__F_STOCK = 4
STONE__E_STOCK = -4
ALL_STRUCTS__CENTER_WORLD = 2
DESTROY_E__E_STRUCT = 2
DESTROY_E__F_STRUCT = 2
DESTROY_E__E_STOCK = 4
DESTROY_E__F_STOCK = 4

# multipliers based on number of adjacent structures
WOOD__F_ADJ = 1.0
WOOD__E_ADJ = -1.0
STONE__F_ADJ = 1.0
STONE__E_ADJ = -1.0
BRIDGE__WATER = 0.5
BRIDGE__BRIDGE = 1.0

# hunger costs, negative is applied in NetLogo
MOVE_HCOST = 1
WATER_MOVE_HCOST = 3
CUT_WOOD_HCOST = 5
MINE_HCOST = 15
BUILD_WOOD_HCOST = 5
BUILD_STONE_HCOST = 10
BUILD_BRIDGE_HCOST = 10
DESTROY_WOOD_COST = 15
DESTROY_STONE_COST = 30
DESTROY_STOCK_COST = 50

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
# format [rock_adjustment, tree_adjustment, bush_adjustment, puddle_adjustment, river_toggle] 
# adjusts the percent of the world area which will be covered by each feature
# tree, and bush: 0 is 10% of green patches, each increase/decrease is a change of 1%
# rocks: 0 is 5% of all patches, each increase/decrease is a change of 0.5%
# puddles: 0 is 1% of all patches, each increase/decrease is a change of 0.5%
# river: 1 is create river, 0 is don't create river
WORLD_TYPES = [[8, 0, 0, 1, 1],
               [6, 3, 3, 3, 1],
               [6, 2, 2, 1, 1],
               [6, 3, 3, 3, 1],
               [6, 2, 2, 1, 1],
               [3, 1, 1, 1, 1],
               [2, 2, 2, 1, 1],
               [8, 0, 0, 1, 1],
               [8, -2, -2, 1, 1],
               [4, 1, 1, -2, 1],
               [3, 1, 1, -1, 1],
               [-5, 1, 5, 1, 1],
               [-10, 1, 1, 1, 1],
               [4, 1, 1, 5, 0],
               [-10, 3, 3, 10, 0]] 


# Not Recomended to Change, If Changed, it's likely neural network architecture will need some tuning
AGENT_SIGHT_RADIUS = 7
TIMESTEPS_IN_OBSERVATIONS = 4


# Dependent on other Variables, just wanted to store it here
MAX_TOTAL_POP = SPECIES_START_POP * NUM_ISLANDS

# Functions for sending config data through PyNetLogo as strings to NetLogo
def get_world_configs(pro_id):
    world_type = WORLD_TYPES[pro_id]
    world_size = [WORLD_WIDTH, WORLD_HEIGHT, PATCH_SIZE]
    world_setup = [BERRY_ABUNDANCE, WOOD_CONCENTRATION, ROCK_DENSITY, STONE_CENTERING]
    team_pop = [SPECIES_START_POP * NUM_SPECIES / 2]
    actor_setup = [STOCKPILE_NUMBER, TEAM_SEPERATION, TEAM_SPLIT_STYLE]
    actor_vars = [START_HUNGER, MAX_EAT, MAX_HUNGER]
    
    return " " + " ".join(map(str,(world_type + world_size + world_setup + team_pop + actor_setup + actor_vars)))

def get_reward_configs():
    rewards1 = [EATING, CUT_TREE, MINE, BUILD_WOOD, BUILD_STONE, BUILD_BRIDGE]
    rewards2 = [DESTROY_E_WOOD, DESTROY_E_STONE, DESTROY_E_STOCKPILE, DESTROY_F]
    wood_b = [WOOD__WATER, WOOD__F_STOCK, WOOD__E_STOCK]
    stone_b = [STONE__WATER, STONE__F_STOCK, STONE__E_STOCK]
    misc_b = [ALL_STRUCTS__CENTER_WORLD]
    dest_b1 = [DESTROY_E__E_STRUCT, DESTROY_E__F_STRUCT]
    dest_b2 = [DESTROY_E__E_STOCK, DESTROY_E__F_STOCK]
    adj_b1 = [WOOD__F_ADJ, WOOD__E_ADJ, STONE__F_ADJ, STONE__E_ADJ]
    adj_b2 = [BRIDGE__WATER, BRIDGE__BRIDGE]
    penalties = [REPEAT_TURN_PENALTY]
    costs1 = [MOVE_HCOST, WATER_MOVE_HCOST, CUT_WOOD_HCOST, MINE_HCOST, BUILD_WOOD_HCOST]
    costs2 = [BUILD_STONE_HCOST, BUILD_BRIDGE_HCOST, DESTROY_WOOD_COST, DESTROY_STONE_COST, DESTROY_STOCK_COST]
    
    return " " + " ".join(map(str,(rewards1 + rewards2 + wood_b + stone_b + misc_b + dest_b1 +
                                   dest_b2 + adj_b1 + adj_b2 + penalties + costs1 + costs2)))



# rpt_turn_penalty
# mv-cost 
# wmv-cost 
# cut-cost 
# mne-cost 
# bld-w-cost 
# bld-s-cost 
# bld-b-cost
# dest-w-cost 
# dest-s-cost 
# dest-stock-cost