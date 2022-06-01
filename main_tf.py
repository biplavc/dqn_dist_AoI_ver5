from collections import UserList
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tqdm
import os
import pickle

import datetime
import copy
import time
from getSNR import *

# from tf_environment import *
# from comet_ml import Experiment

# experiment = Experiment("HsbMT2nT816RPUXC1LLkVvEe0")

now = datetime.datetime.now()


# from tf_c51 import *
# from tf_sac import *
# from mrb_scheduling import *
# from rr_scheduling import * ## has worse performance than random, so made a new round robin that does better
# from tf_reinforce import *
# from path_loss_probability import *

from create_graph_1 import *
import itertools
from itertools import product  
from tf_dqn import *
from random_scheduling import *
from greedy_scheduling import *
from mad_scheduling import *
# from lrb_new import *
from rr_scheduling import *
from omad_links_UL_scheduling import *
from omad_greedy_UL_scheduling import *
from omad_cumAoI_UL_scheduling import *
from pf_scheduling import *
from laf_scheduling import *
from lad_scheduling import *
from new_scheduling import *

import sys

from joblib import Parallel, delayed
import multiprocessing as mp

from parameters import *

random.seed(4)
# tf.random.set_seed(42)

print_matrix = True

def distributed_run(arguments):
  
    print(f"passed arguments are {arguments}\n", file = open(folder_name + "/results.txt", "a"), flush = True)
    print(f"passed arguments are {arguments}")

    # pool.starmap(do_scheduling, [(arg[0], arg[1], arg[2]) for arg in arguments]) ## this enable multiprocessing but I am getting memroy allocation and other CUDA related errors with this, so now using sequential execution
    
    for j in arguments: ## no multiprocessing
        do_scheduling(j[0],j[1],j[2])
    
#############################################################

def do_scheduling(deployment, I, scheduler):
    
    STEPS = list(range(min_steps, min_steps+1, interval)) ## will just run once 
    
    for T in STEPS:
        print(f"\nsimulation will run for T={T} steps\n")
        global print_matrix
        
        deployment_options = ["MDS", "RP"]
        scheduler_options  = ["random", "greedy", "mad", "omad_greedy_UL", "omad_cumAoI_UL", "omad_links_UL" , "rr", "dqn", "pf", "laf", "lad", "new"] ## "random", "greedy", "mad", "omad_greedy_UL", "omad_cumAoI_UL", "omad_links_UL" , "rr", "dqn", "pf"
        assert(deployment in deployment_options and scheduler in scheduler_options)
        # schedulers  = ["dqn" "random", "greedy", "mad", "omad_greedy_UL", "rr", "pf"]

        random.seed(4) ## this seed ensures same location of users in every case, keep both seeds
        np.random.seed(4)
        
        if test_case:

            drones_needed           = 1
            users_per_drone         = [I]
            
            adj_matrix = np.random.randint(2, size=(I, I))
            for i in range(I):
                for j in range(I):
                    if i==j:
                        adj_matrix[i][j] = 0
   
            # adj_matrix              = np.array([[0, 1, 1, 0, 0], ## 5 UL 10 DL
            #                                     [0, 0, 1, 1, 0],
            #                                     [0, 0, 0, 1, 1],
            #                                     [1, 0, 0, 0, 1],
            #                                     [1, 1, 0, 0, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 1, 0, 0], ## 5 UL 8 DL
            #                                     [0, 0, 1, 1, 0],
            #                                     [0, 0, 0, 0, 1],
            #                                     [1, 0, 0, 0, 1],
            #                                     [1, 0, 0, 0, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 1, 1, 1], ## 5 UL 11 DL
            #                                     [1, 0, 0, 1, 1],
            #                                     [0, 0, 0, 1, 0],
            #                                     [1, 0, 0, 0, 1],
            #                                     [1, 1, 0, 0, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 0, 0, 0], ## 5 UL 5 DL
            #                                     [0, 0, 1, 0, 0],
            #                                     [0, 0, 0, 1, 0],
            #                                     [0, 0, 0, 0, 1],
            #                                     [1, 0, 0, 0, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 1], ## one to all
            #                                     [1, 0, 1],
            #                                     [1, 1, 0]])
            
            # adj_matrix              = np.array([[0, 1, 0], ## one to one
            #                                     [0, 0, 1],
            #                                     [1, 0, 0]])

            # adj_matrix              = np.array([[0, 1, 1], ## varying
            #                                     [0, 0, 1],
            #                                     [1, 1, 0]])
            
            
            # adj_matrix              = np.array([[0, 1], ## 2 UL 2 DL
            #                                     [1, 0]])
            
            
            # adj_matrix              = np.array([[0, 1, 1, 0], ## 4 UL 8 DL
            #                                     [0, 0, 1, 1],
            #                                     [1, 0, 0, 1],
            #                                     [1, 1, 0, 0]])

            
            # adj_matrix              = np.array([[0, 1, 1, 0], ## 4 UL 7 DL
            #                                     [0, 0, 1, 1],
            #                                     [1, 0, 0, 1],
            #                                     [0, 1, 0, 0]])
            
            # adj_matrix              = np.array([[0, 0, 1, 0], ## 4 UL 4 DL
            #                                     [0, 0, 0, 1],
            #                                     [1, 0, 0, 0],
            #                                     [0, 1, 0, 0]])
            
            if print_matrix == True:
                print(f"adj_matrix = \n\n\n", np.matrix(adj_matrix))
                print_matrix = True
            
            tx_rx_pairs = []
            tx_users    = []
            
            rows, columns = np.shape(adj_matrix)
            # print(f"rows = {rows}, columns = {columns}")
            
            ## relevant pair calculation starts
            
            # age at the final dest will be w.r.t only these pairs.  
            for i in range(rows):
                for ii in range(columns):
                    if adj_matrix[i,ii]==1:
                        pair = [i + 10, ii + 10] ## 10 as count is 10 from main_tf.py where user IDs start from 10
                        tx_rx_pairs.append(pair)
            
            for i in tx_rx_pairs:
                if i[0] not in tx_users:
                    tx_users.append(i[0])
                    
            assert drones_needed    ==len(users_per_drone)
            
            drones_coverage         = []
            
            count = 10 # user IDs will start from this. and this also ensured that UAV and users have different IDs. Ensure number of UAVs is less than the count
            for i in range(drones_needed):
                individual_drone_coverage = [x for x in range(count, count + users_per_drone[i])]
                count = individual_drone_coverage[-1] + 1
                drones_coverage.append(individual_drone_coverage)
                
            user_list = []
            UAV_list = np.arange(drones_needed)
            for i in drones_coverage:
                for j in i:
                    if j!=0: ## user will not contain 0
                        user_list.append(j)
                        
            RB_needed_UL = {x : random.choice([1]) for x in tx_users}
            # RB_needed_UL = {x : random.choice([1]) for x in user_list}
            RB_needed_DL = {tuple(x) : RB_needed_UL[x[0]] for x in tx_rx_pairs}


            assert (max(user_list) - min(user_list))+1 == sum(users_per_drone)
            # time.sleep(10)

                        
            if periodic_generation:
                periodicity = {x:random.choice([1,2,3]) for x in user_list} # biplav {10:1,11:2,12:3,13:2,14:3} #
            else:
                periodicity = {x:1 for x in user_list}
            
            I = len(user_list) # changed to the needed value
            
            L = 1_000 # length of the area
            B = 1_000 # breadth of the area
            
            BS_location = [L/2, B/2]
            
            user_locations = {}
            
            distances = []
            
            for j in user_list:
                x = np.round(random.uniform(0, L),2)
                y = np.round(random.uniform(0, B),2)
                user_locations[j] = [x,y]
                distances.append(math.sqrt((BS_location[0]-x)**2 + (BS_location[1]-y)))
                
            print(f"BS_location = {BS_location}, user_locations = {user_locations}, max distance = {np.max(distances)}, min distance = {np.min(distances)}, no of senders = {len(tx_users)}, no of sender-receiver pairs = {len(tx_rx_pairs)}")
            

            if packet_loss == True:
                SNR_threshold = 3 ## dB from URLLC MCS5 QPSK

            else:
                SNR_threshold = -1_000_000
                
            packet_download_loss_thresh  = {tuple(yy) : SNR_threshold for yy in tx_rx_pairs}
            packet_upload_loss_thresh  = {yy : SNR_threshold for yy in user_list}

                
                
        else: ## user defined UAV and user configuration
            
            assert test_case == True, "Test Case cannot be false here" # denominator can't be 0 
                
        print(f'UAV_list = {UAV_list}, drones_coverage = {drones_coverage}, user_list = {user_list}, periodicity = {periodicity}, RB_total_UL = {RB_total_UL}, RB_total_DL = {RB_total_DL}, total_RB_needed_UL = {np.sum(list(RB_needed_UL.values()))}, total_RB_needed_DL = {np.sum(list(RB_needed_DL.values()))}, RB_needed_UL = {RB_needed_UL}, RB_needed_DL = {RB_needed_DL} for {deployment} deployment for {I} users under {scheduler} scheduling, packet_download_loss_thresh = {packet_download_loss_thresh}, packet_upload_loss_thresh = {packet_upload_loss_thresh}, SNR_threshold = {SNR_threshold}\n', file=open(folder_name + "/results.txt", "a"), flush=True)  

        str_x = str(deployment) + " placement with " + str(I) + " users and " + str(T) + " slots needs " + str(scheduler) + " scheduler and "  + str(drones_needed) + " drones\n"
        print(f'{str_x}', file=open(folder_name + "/drones.txt", "a"), flush=True)
        
        if scheduler == "greedy":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            greedy_overall[I], greedy_final[I], greedy_overall_times[I] = greedy_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)  
            t2 = time.time()
            print("greedy for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(greedy_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_greedy_overall.pickle", "wb")) 
            pickle.dump(greedy_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_greedy_final.pickle", "wb"))
            pickle.dump(greedy_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "_greedy_all_actions.pickle", "wb")) 

        
        if scheduler == "random":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            random_overall[I], random_final[I], random_overall_times[I] = random_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("random for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(random_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_overall.pickle", "wb")) 
            pickle.dump(random_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_final.pickle", "wb")) 
            pickle.dump(random_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_random_all_actions.pickle", "wb")) 
            
        if scheduler == "omad_links_UL":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            omad_links_UL_overall[I], omad_links_UL_final[I], omad_links_UL_overall_times[I] = omad_links_UL_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("omad_links_UL for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(omad_links_UL_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_links_UL_overall.pickle", "wb")) 
            pickle.dump(omad_links_UL_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_links_UL_final.pickle", "wb")) 
            # pickle.dump(omad_links_UL_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_all_actions.pickle", "wb"))
            
        if scheduler == "omad_cumAoI_UL":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            omad_cumAoI_UL_overall[I], omad_cumAoI_UL_final[I], omad_cumAoI_UL_overall_times[I] = omad_cumAoI_UL_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("omad_cumAoI_UL for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(omad_cumAoI_UL_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_cumAoI_UL_overall.pickle", "wb")) 
            pickle.dump(omad_cumAoI_UL_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_cumAoI_UL_final.pickle", "wb")) 
            # pickle.dump(omad_cumAoI_UL_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_waoi_all_actions.pickle", "wb"))
            
        if scheduler == "omad_greedy_UL":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            omad_greedy_UL_overall[I], omad_greedy_UL_final[I], omad_greedy_UL_overall_times[I] = omad_greedy_UL_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("omad_greedy_UL for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(omad_greedy_UL_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_greedy_UL_overall.pickle", "wb")) 
            pickle.dump(omad_greedy_UL_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_greedy_UL_final.pickle", "wb")) 
            # pickle.dump(omad_greedy_UL_overall_times, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_waoi_all_actions.pickle", "wb"))
            
            
        if scheduler == "mad":
            t1 = time.time()
            #with tf.device('/CPU:0'):
            mad_overall[I], mad_final[I], mad_overall_times[I] = mad_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()            
            print("MAD for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(mad_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_overall.pickle", "wb")) 
            pickle.dump(mad_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_final.pickle", "wb"))
            pickle.dump(mad_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_mad_all_actions.pickle", "wb")) 
            
        
        if scheduler == "dqn":
            t1 = time.time()
            dqn_overall[I], dqn_final[I], dqn_overall_times[I] = tf_dqn(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("DQN for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(dqn_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_overall.pickle", "wb")) 
            pickle.dump(dqn_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_final.pickle", "wb"))
            pickle.dump(dqn_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_all_actions.pickle", "wb"))


        if scheduler == "rr":
            t1 = time.time()
            rr_overall[I], rr_final[I], rr_overall_times[I] = rr_new_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("RR for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(rr_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_rr_overall.pickle", "wb")) 
            pickle.dump(rr_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_rr_final.pickle", "wb"))
            pickle.dump(rr_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_rr_all_actions.pickle", "wb"))
            
        if scheduler == "pf":
            t1 = time.time()
            pf_overall[I], pf_final[I], pf_overall_times[I] = pf_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("PF for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(pf_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_pf_overall.pickle", "wb")) 
            pickle.dump(pf_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_pf_final.pickle", "wb"))
            pickle.dump(pf_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_pf_all_actions.pickle", "wb"))

        if scheduler == "laf":
            t1 = time.time()
            laf_overall[I], laf_final[I], laf_overall_times[I] = laf_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("LAF for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(laf_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_laf_scheduling_overall.pickle", "wb")) 
            pickle.dump(laf_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_laf_scheduling_final.pickle", "wb"))
            pickle.dump(laf_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_laf_scheduling_all_actions.pickle", "wb"))

        if scheduler == "lad":
            t1 = time.time()
            lad_overall[I], lad_final[I], lad_overall_times[I] = lad_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("LAD for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(lad_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_lad_scheduling_overall.pickle", "wb")) 
            pickle.dump(lad_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_lad_scheduling_final.pickle", "wb"))
            pickle.dump(lad_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_lad_scheduling_all_actions.pickle", "wb"))

        if scheduler == "new":
            t1 = time.time()
            new_overall[I], new_final[I], new_overall_times[I] = new_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
            t2 = time.time()
            print("new for ", I, " users and T = ", T, " slots took ", t2-t1, " seconds to complete", file=open(folder_name + "/results.txt", "a"), flush=True)
            pickle.dump(new_overall, open(folder_name + "/" + deployment + "/" + str(I) + "U_new_scheduling_overall.pickle", "wb")) 
            pickle.dump(new_final, open(folder_name + "/" + deployment + "/" + str(I) + "U_new_scheduling_final.pickle", "wb"))
            pickle.dump(new_all_actions, open(folder_name + "/" + deployment + "/" + str(I) + "U_new_scheduling_all_actions.pickle", "wb"))


#############################################################
    
if __name__ == '__main__':
        

    now_str_1 = now.strftime("%Y-%m-%d %H:%M")
    folder_name = 'models/' +  now_str_1
    
    # print(f"\n\nSTATUS OF GPU : {tf.test.is_built_with_gpu_support() and {tf.test.is_gpu_available()}}\n\n", file = open(folder_name + "/results.txt", "a"), flush = True)
    
    # print(f"\n\nSTATUS OF GPU : {tf.test.is_built_with_gpu_support() and {tf.test.is_gpu_available()}}\n\n")
    
    folder_name_MDS = folder_name + "/MDS"
    folder_name_random = folder_name + "/RP" ## RP means random placement

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
        os.makedirs(folder_name_MDS)
        os.makedirs(folder_name_random) 
        
    print("execution started at ", now_str_1, file = open(folder_name + "/results.txt", "a"), flush = True)

    deployments = ["RP"] 
    
    schedulers  = ["new", "mad"]  
    # "random", "greedy", "mad", "omad_greedy_UL", "rr", "pf", "dqn", "laf"

    limit_memory = True ## enabling this makes the code not being able to find CUDA device
    
#############################

    test_case = bool
    packet_loss = bool
    periodic_generation = bool
    
    experiment = 1 ## 1 or 5 ## biplav

    if experiment == 1:
        test_case           = True
        packet_loss         = False
        periodic_generation = False

    elif experiment == 2:
        test_case           = True
        packet_loss         = True
        periodic_generation = False
        
    elif experiment == 3:
        test_case           = False
        packet_loss         = False
        periodic_generation = False

    elif experiment == 4:
        test_case           = False
        packet_loss         = True
        periodic_generation = False

    elif experiment == 5:
        test_case           = True
        packet_loss         = False
        periodic_generation = True

    elif experiment == 6:
        test_case           = True
        packet_loss         = True
        periodic_generation = True
        
    elif experiment == 7:
        test_case           = False
        packet_loss         = False
        periodic_generation = True

    elif experiment == 8:
        test_case           = False
        packet_loss         = True
        periodic_generation = True
    

    if test_case:
        users = [12] ##biplav
    else:
        users = [3]

#############################
    
    arguments = list(itertools.product(deployments, users, schedulers)) ## deployment, I, scheduler
    
    dqn_overall = {}
    dqn_final = {}
    dqn_all_actions = {}
    dqn_overall_times = {} # dqn_overall_avg for each MAX_STEPS
    
    c51_overall = {}
    c51_final = {}
    c51_all_actions = {}
    c51_overall_times = {} # c51_overall_avg for each MAX_STEPS
    
    reinforce_overall = {}
    reinforce_final = {}
    reinforce_all_actions = {}
    reinforce_overall_times = {} # reinforce_overall_avg for each MAX_STEPS
    
    random_overall = {} ## sum of age at destination nodes for all of the MAX_STEPS time steps
    random_overall = {} ## sum of age at destination nodes for all of the MAX_STEPS time steps
    random_final   = {} ## sum of age at destination nodes for step =  MAX_STEPS i.e. last time step
    random_all_actions = {}
    random_overall_times = {} # random_overall_times for each MAX_STEPS
    
    greedy_overall = {}
    greedy_final   = {}
    greedy_all_actions = {}
    greedy_overall_times = {} # greedy_overall_times for each MAX_STEPS
    
    mad_overall = {}
    mad_final   = {}
    mad_all_actions = {}
    mad_overall_times = {} #  mad_overall_avg for each MAX_STEPS
    
    sac_overall = {}
    sac_final   = {}
    sac_all_actions = {}
    sac_overall_times = {} # sac_overall_avg for each MAX_STEPS
    
    lrb_overall = {}
    lrb_final   = {}
    lrb_all_actions = {}
    lrb_overall_times = {} # lrb_overall_avg for each MAX_STEPS

    mrb_overall = {}
    mrb_final   = {}
    mrb_all_actions = {}
    mrb_overall_times = {} # mrb_overall_avg for each MAX_STEPS
    
    rr_overall = {}
    rr_final   = {}
    rr_all_actions = {}
    rr_overall_times = {} # rr_overall_avg for each MAX_STEPS
    
    pf_overall = {}
    pf_final   = {}
    pf_all_actions = {}
    pf_overall_times = {} # rr_overall_avg for each MAX_STEPS
    
    new_overall = {}
    new_final   = {}
    new_all_actions = {}
    new_overall_times = {} # rr_overall_avg for each MAX_STEPS
    
    laf_overall = {}
    laf_final   = {}
    laf_all_actions = {}
    laf_overall_times = {} # rr_overall_avg for each MAX_STEPS 
    
    lad_overall = {}
    lad_final   = {}
    lad_all_actions = {}
    lad_overall_times = {} # rr_overall_avg for each MAX_STEPS     
    
    omad_greedy_UL_overall = {} 
    omad_greedy_UL_final = {}
    omad_greedy_UL_all_actions = {}
    omad_greedy_UL_overall_times = {} # omad_greedy_UL_overall_times for each MAX_STEPS
    
    omad_cumAoI_UL_overall = {}
    omad_cumAoI_UL_final = {}
    omad_cumAoI_UL_all_actions = {}
    omad_cumAoI_UL_overall_times = {} # omad_cumAoI_UL_overall_times for each MAX_STEPS
    
    omad_links_UL_overall = {}
    omad_links_UL_final = {} 
    omad_links_UL_all_actions = {}
    omad_links_UL_overall_times = {}



    pool = mp.Pool(mp.cpu_count())
    print(f"pool is {pool} \n\n", file = open(folder_name + "/results.txt", "a"))


    print(f"num_iterations = {num_iterations}, random_episodes = {random_episodes}, min_steps = {min_steps}, gamma = {set_gamma}, learning_rate = {learning_rate}, fc_layer_params = {fc_layer_params}, replay_buffer_capacity = {replay_buffer_capacity}, log_interval = {log_interval}, log_interval_random = {log_interval_random} \n\n",  file = open(folder_name + "/results.txt", "a"), flush = True)    

    print(f"experiment is {experiment} with test_case = {test_case}, packet_loss = {packet_loss}, periodic_generation = {periodic_generation}", file = open(folder_name + "/results.txt", "a"))
    
    print(f"experiment is {experiment} with test_case = {test_case}, packet_loss = {packet_loss}, periodic_generation = {periodic_generation}")

    distributed_run(arguments)
    pool.close()    
