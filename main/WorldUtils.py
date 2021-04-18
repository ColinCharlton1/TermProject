# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:12:45 2021

@author: Colin
"""

import numpy as np
from numba import njit
from math import ceil

# Simple adaptation of population distribution described in: 
# Leibo, Joel Z., et al. "Malthusian reinforcement learning." arXiv preprint arXiv:1812.07019 (2018).
def distribute_populations(max_population, rng, current_dist_probs, gen_rewards):
    num_islands = current_dist_probs.shape[1]
    newdist = np.ones((current_dist_probs.shape[1], current_dist_probs.shape[0]), dtype=np.int16)
    for species in range(current_dist_probs.shape[0]):
        if gen_rewards[:, species].min() < 0:
            gen_rewards[:, species] += abs(gen_rewards[:, species].min() * 2)
        new_species_probs = gen_rewards[:, species] / gen_rewards[:, species].sum()
        current_dist_probs[species] = (current_dist_probs[species] * 0.95 + new_species_probs * 0.05)
        
        new_species_dist = rng.choice(num_islands, p = current_dist_probs[species], size=max_population - num_islands)
        for ix in new_species_dist:
            newdist[ix,species] += 1
    return newdist

def get_species_dist_string(dist):
    result = " (list "
    for d in range(int(len(dist) / 2)):
        result += str(dist[d]) + " "
    result += ") (list "
    for d in range(int(len(dist) / 2), len(dist)):
        result += str(dist[d]) + " "
    result += ")"
    return result

@njit()
def update_world(target_world, updates, asr):
    for x, y, feature, val in updates:
        target_world[y + asr,
                    x + asr,
                    feature] = val    
        
@njit()
def get_masked_world_data(target_world, species_vals, asr):
    return [mask_vision(target_world[int(y) : int(y) + 2 * asr + 1,
                                     int(x) : int(x) + 2 * asr + 1 ])
                                     for x, y in species_vals[:,0:2]]
     
@njit()
def get_world_data(target_world, species_vals, asr):
    return [target_world[int(y) : int(y) + 2 * asr + 1,
                         int(x) : int(x) + 2 * asr + 1 ]
                         for x, y in species_vals[:,0:2]]

# numba was incredibly helpful in speeding up the way I decided to do masking
# it would take an unreasonable amount of time without it
@njit()
def mask_vision(start_matrix, max_obstruction = 0.9):
    my_matrix = start_matrix.copy()
    mask = np.zeros(my_matrix.shape[:-1])
    center = int(mask.shape[0] / 2)
    for offset in range(1, center + 1):
        if (my_matrix[center,center+offset,-1] < 9 and my_matrix[center,center+offset,-1] > 0):
            mask_axises(my_matrix, mask, center, max_obstruction, xoffset=offset)
        if (my_matrix[center,center-offset,-1] < 9 and my_matrix[center,center+offset,-1] > 0):
            mask_axises(my_matrix, mask, center, max_obstruction, xoffset=-offset)
        if (my_matrix[center+offset,center,-1] < 9 and my_matrix[center,center+offset,-1] > 0):
            mask_axises(my_matrix, mask, center, max_obstruction, yoffset=offset)
        if (my_matrix[center-offset,center,-1] < 9 and my_matrix[center,center+offset,-1] > 0):
            mask_axises(my_matrix, mask, center, max_obstruction, yoffset=-offset)
            
    for dx in range(1,center+1):
        for dy in range(1,center+1):
            if (my_matrix[center - dy, center + dx, -1] < 9 and my_matrix[center,center+offset,-1] > 0):
                mask_tiles(my_matrix, mask, center, max_obstruction, dx, -dy)
            if (my_matrix[center - dy, center - dx, -1] < 9 and my_matrix[center,center+offset,-1] > 0):
                mask_tiles(my_matrix, mask, center, max_obstruction, -dx, -dy)
            if (my_matrix[center + dy, center + dx, -1] < 9 and my_matrix[center,center+offset,-1] > 0):
                mask_tiles(my_matrix, mask, center, max_obstruction, dx, dy)
            if (my_matrix[center + dy, center - dx, -1] < 9 and my_matrix[center,center+offset,-1] > 0):
                mask_tiles(my_matrix, mask, center, max_obstruction, -dx, dy)
    return my_matrix
 
@njit()
def mask_tiles(matrix, mask, center, max_obstruction, dx=0, dy=0):                        
    upper_slope = (2 * abs(dy) + 1) / (2 * abs(dx) - 1)
    lower_slope = (2 * abs(dy) - 1) / (2 * abs(dx) + 1)
    xsign = 1 if dx == abs(dx) else -1
    ysign = 1 if dy == abs(dy) else -1
    dx = abs(dx)
    dy = abs(dy)
    for x in range(0, center-dx):
        x_ystart = x * lower_slope
        for y in range(0, min(ceil(lower_slope * x) + 1, center-dy+1)):
            x1 = min( x + 1, max(x, y / lower_slope))
            x2 = min(x1 + 1 / lower_slope, x + 1)
            y1 = x1 * lower_slope
            tri_area = (lower_slope / 2) * (x2*x2 - x1*x1) - x_ystart * (x2 - x1)
            xy_area = tri_area + (x + 1 - x2) + ((x2 - x1) * min(y1 - y, 1))
            if (1 - xy_area >= max_obstruction and matrix[center+(dy+y)*ysign, center+(dx+x+1)*xsign,-1] > 8):
                matrix[center+(dy+y)*ysign, center+(dx+x+1)*xsign] = 0
            elif (1 - xy_area >= 1):
                matrix[center+(dy+y)*ysign, center+(dx+x+1)*xsign] = 0
            elif (1 - xy_area > 0):
                mask[center+(dy+y)*ysign, center+(dx+x+1)*xsign] += 1 - xy_area
        
    for x in range(0, center-dx+1):
        x_ystart = x * upper_slope
        for y in range(0, center-dy):
            if (y < ceil((x-1) * lower_slope)): continue
            x1 = min( x + 1, max(x, y / upper_slope))
            x2 = min(x1 + 1 / upper_slope, x + 1)
            y1 = x1 * upper_slope
            tri_area = (upper_slope / 2) * (x2*x2 - x1*x1) - x_ystart * (x2 - x1)
            xy_area = tri_area + (x + 1 - x2) + ((x2 - x1) * min(y1 - y, 1))
            if (xy_area >= max_obstruction and matrix[center+(dy+y+1)*ysign, center+(dx+x)*xsign,-1] > 8):
                matrix[center+(dy+y+1)*ysign, center+(dx+x)*xsign] = 0
            elif (xy_area >= 1):
                matrix[center+(dy+y+1)*ysign, center+(dx+x)*xsign] = 0
            elif (xy_area > 0):
                mask[center+(dy+y+1)*ysign, center+(dx+x)*xsign] += xy_area
    for y in range(len(mask[0])):
        for x in range(len(mask[0])):
            if mask[y,x] > max_obstruction and matrix[y, x, -1] > 8:
                matrix[y, x] = 0
            elif  mask[y,x] >= 1:
                matrix[y, x] = 0
                    
@njit()
def mask_axises(matrix, mask, center, max_obstruction, xoffset=0, yoffset=0):                        
    slope = 1 / max((2 * abs(xoffset + yoffset) - 1), 1)
    xsign = 1 if xoffset == abs(xoffset) else -1
    ysign = 1 if yoffset == abs(yoffset) else -1
    for x in range(1, center-abs(xoffset + yoffset)+1):
        x_ystart = x * slope
        if(xoffset != 0): matrix[center, center+xoffset+x*xsign] = 0
        if(yoffset != 0): matrix[center+yoffset+x*ysign, center] = 0
        for y in range(0, min(ceil(slope * (x + 1)), center)):
            x1 = min(x + 1, max(x, y / slope))
            x2 = min(x1 + 1 / slope, x + 1)
            y1 = x1 * slope
            tri_area = (slope / 2) * (x2*x2 - x1*x1) - x_ystart * (x2 - x1)
            xy_area = tri_area + (x + 1 - x2) + ((x2 - x1) * min(y1 - y, 1))
            if (xy_area >= 1 and xoffset != 0):
                matrix[center+y+1, center+xoffset+x*xsign] = 0
                matrix[center-y-1, center+xoffset+x*xsign] = 0
            elif (xy_area >= 1):
                matrix[center+yoffset+x*ysign, center+y+1] = 0
                matrix[center+yoffset+x*ysign, center-y-1] = 0
            elif (xy_area > 0 and xoffset != 0):
                if(mask[center+y+1, center+xoffset+x*xsign] + xy_area > 1): 
                    mask[center+y+1, center+xoffset+x*xsign] = max(xy_area, mask[center+y+1, center+xoffset+x*xsign])
                else:
                    mask[center+y+1, center+xoffset+x*xsign] += xy_area
                if(mask[center+y-1, center+xoffset+x*xsign] + xy_area > 1): 
                    mask[center-y-1, center+xoffset+x*xsign] = max(xy_area, mask[center-y-1, center+xoffset+x*xsign])
                else:
                    mask[center-y-1, center+xoffset+x*xsign] += xy_area   
            elif (xy_area > 0):
                if(mask[center+yoffset+x*ysign, center+y+1] + xy_area > 1): 
                    mask[center+yoffset+x*ysign, center+y+1] = max(xy_area, mask[center+yoffset+x*ysign, center+y+1])
                else:
                    mask[center+yoffset+x*ysign, center+y+1] += xy_area
                if(mask[center+yoffset+x*ysign, center-y-1] + xy_area > 1): 
                    mask[center+yoffset+x*ysign, center-y-1] = max(xy_area, mask[center+yoffset+x*ysign, center-y-1])
                else:
                    mask[center+yoffset+x*ysign, center-y-1] += xy_area
