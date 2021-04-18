# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:10:47 2021

@author: Colin
"""

import os
import numpy as np
import pandas as pd
import pyNetLogo
import queue
from WorldUtils import get_species_dist_string
import ConfigurationManager as cf
import time
from multiprocessing import Process, Queue
from DataManager import DataManager
from WorldUtils import update_world, get_world_data, get_masked_world_data, distribute_populations

class NetLogoProcessManager():
    def __init__(self, start_pop_dist):
        self.actionsDirName = 'actions'
        if not os.path.exists(self.actionsDirName):
            os.mkdir(self.actionsDirName)
        self.inques = []
        lockque = Queue()
        self.pros = []
        
        outque = Queue()
        for __ in range(cf.NUM_PROCESSES):
                lockque.put(1)
        for num in range(cf.NUM_ISLANDS):
            inque = Queue()
            self.inques.append(inque)
            p = Process(target=netLogo_instance, args=(
                lockque, inque, outque, num, start_pop_dist[num], self.actionsDirName))
            self.pros.append(p)
            p.start()
            # NetLogo Startup is threaded and very resource intensive,
            # this gives it a chance to start in each process without bottlenecking the CPU
            # might need more time if using even more threads
            time.sleep(3)
     
    def get_initial_worlds(self):
        for inque in self.inques:
            inque.put(-1)
            
    def get_world_update(self):
        for inque in self.inques:
            inque.put(0)
    
    def kill_ilsand(self, pro_id):
        self.inques[pro_id].put(3)
        
    def send_actions(self, pro_id, ids, actions):
        df = pd.DataFrame(np.stack([ids, actions], axis=1), columns=["who", "action"])
        df.to_csv(self.actionsDirName + "/actions" + str(pro_id) + ".csv")
        self.inques[pro_id].put(1)
        
    def wrap_up_generation(self, new_pop_dist):
        for index, inque in enumerate(self.inques):
            inque.put(2)
            inque.put(new_pop_dist[index])




# all interactions with NetLogo are adapted from example code shared by the creators of PyNetLogo
# PyNetLogo website: https://pynetlogo.readthedocs.io/en/latest/index.html
# code available at: https://github.com/quaquel/pyNetLogo
# Reads signals from inque as follows:
# -1: get start array of new world
# 0 : report data to main
# 1 : execute tick
# 2 : wrap up generation
# 3 : mark process as having no actors left alive
# 4 : shut down process
def netLogo_instance(mylockque, myinque, myoutque, id_num, start_dist, actionsDirName):
    lock = mylockque.get()  # wait for turn to run startup
    print("hello from process {}".format(os.getpid()))
    dead = False
    if id_num == 0:
        netlogo = pyNetLogo.NetLogoLink(gui=True)
    else:
        netlogo = pyNetLogo.NetLogoLink(gui=False)
        
    netlogo.load_model("C:/Users/Colin/Desktop/Winter2021/CPSC565/TermProject/repo/TeamBuilderSim.nlogo")    
    
    netlogo.command("set-env-params " + cf.get_world_configs(id_num))
    netlogo.command("set-reward-params" + cf.get_reward_configs())
    netlogo.command("setup " + get_species_dist_string(start_dist))
    print("process {} finished setup".format(os.getpid()))
    mylockque.put(lock)
    lock = 0
    while True:
        # This is not a good example of que thread safety, but better than nothing
        try:
            signal = myinque.get(timeout=600)
        except queue.Empty:
            break
        if lock == 0:
            try:
                lock = mylockque.get(timeout=600)
            except queue.Empty:
                break
            
        if signal == -1:
            vegetation = netlogo.report("env-vegetation")
            actors = netlogo.report("env-actors")
            color = netlogo.report("env-types")
            new_world = np.column_stack((vegetation, actors, color)).clip(-25, 25).astype(np.int8)
            myoutque.put((id_num, new_world))
            mylockque.put(lock)
            lock = 0
            
        if signal == 0:
            if not dead:
                myids = netlogo.report("get-ids")
                if type(myids) is np.ndarray:
                    updates = netlogo.report("get-updates")
                    new_species_vals = netlogo.report("get-actor-data")
                    myoutque.put((id_num, myids.astype(np.int16), updates, new_species_vals))
                else:
                    myoutque.put((id_num, myids, updates, new_species_vals))
            mylockque.put(lock)
            lock = 0
            
        elif signal == 1:
            my_df = pd.read_csv(actionsDirName + "/actions" + str(id_num) + ".csv", header=0)
            netlogo.write_NetLogo_attriblist(my_df[["who", "action"]], "actor")
            netlogo.command("end-step")
            mylockque.put(lock)
            lock = 0
            
        elif signal == 2:
            myoutque.put((id_num, netlogo.report("get-actor-stats"), netlogo.report("get-stockpiles")))
            new_dist = myinque.get()
            netlogo.command("setup " + " " + get_species_dist_string(new_dist))
            dead = False
            mylockque.put(lock)
            lock = 0
            
        elif signal == 3:
            dead = True
            mylockque.put(lock)
            lock = 0
            
        elif signal == 4:
            break
        
    netlogo.kill_workspace()
    mylockque.put(lock)
    lock = 0
    return