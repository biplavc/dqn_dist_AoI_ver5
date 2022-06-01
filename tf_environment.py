from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from getSNR import getSNR
import tensorflow as tf
import numpy as np

import random
from itertools import combinations

# from path_loss_probability import *
# from age_calculation import *
# from drl import *

from itertools import product  
import itertools
from create_graph_1 import *

import time
import math
import scipy
import pickle
import matplotlib
import matplotlib.pyplot as plt
import copy
import base64
# import imageio

import matplotlib.pyplot as plt
import os
# import reverb
import tempfile
import functools
import operator

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network


tf.compat.v1.enable_v2_behavior()

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import triggers
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import strategy_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.policies import policy_saver
from tf_agents.metrics import py_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

## sac
# from tf_agents.train import actor
# from tf_agents.train import learner
# from tf_agents.train import triggers
# from tf_agents.train.utils import spec_utils
# from tf_agents.train.utils import strategy_utils
# from tf_agents.train.utils import train_utils
## sac ends

tempdir = tempfile.gettempdir()
random.seed(42)
# tf.random.set_seed(42)

################## PARAMETERS NEEDED TO BE CONFIGURED EVERY RUN

## shell files ntasks = |users| * |scheduler| * |placement| = 4*6*2 = 48, use 4 nodes, in scheduler only consider the ML ones. greedy mad and random are done in less than a minute
limit_memory = True
verbose = False
# verbose = True
comet = False
# comet = True
CSI_as_state = False
sample_error_in_CSI = False ## has no meaning if CSI_as_state=False; if CSI_as_state True, this True will mean only UAV CSI is included in state
time_in_state = False


min_steps = 5
interval  = 1

delay_include = 0 # 0 for slot and 1 to include it

modulation_index  = [0,1,2,3] # simple index to get the index of modulation_index
packet_size       = 64 # bits
modulation_orders = [2,4,6,8]
base_throughput = 1.07 # in Mbps
throughputs = [base_throughput, 2*base_throughput, 3*base_throughput, 4*base_throughput]
np.set_printoptions(precision=2) # https://stackoverflow.com/questions/12439753/set-global-output-precision-python/38447064


MAX_AGE = min_steps + 1 + 1 # 20

coverage_capacity = 3 # max users 1 UAV can cover, used in create_graph_1

set_gamma = 1
RB_total_UL = 6 # L R_u, sample. has to be less than number of tx_users
RB_total_DL = 14 # K R_d, update. has to be less than number of tx_rx_pairs

#@param {type:"integer"} # number of times collect_data is called, log_interval and eval_interval are used here. number of times the collect_episodes(...) will run. each collect_episode(...) runs for collect_episodes_per_iteration episodes to fill the buffer. once one iteration is over, the train_env is run on it and then buffer is clear. This value doesn't add to the returns that is showed as the final performance.

random_episodes = 100 # all schedulers except dqn runs this number of times
log_interval_random = 1 if (random_episodes//10==0) else random_episodes//10

num_iterations = 1_000_000 # 1_000_000 # dqn runs this number of times
log_interval = 10_000 # @param {type:"integer"} # how frequently to print out in console


# if packet failure occurs independently at each user-UAV and UAV_BS link, the greedy will still keep targetting the most aged user and therefore greedy will still perform well which can be seen in Optimal Scheduling Policy for Minimizing Age of Information with a Relay. Eg with a failure rate 0.5, greedy will keep targetting the most aged which might still fail and it could have been able to reduce the age by targetting some other user. Not sure what will happen with failure rates, lets see.

fc_layer_params = (1024, 1024) # for 5 user 2 SC (32,16) # for 3 user 1 SC


################### DON'T CHANGE THESE    

## https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0 ## remove this line for ARC, keep for dgx2-1

## https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"GPUs are {gpus}")
if limit_memory:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
# https://www.tensorflow.org/guide/gpu
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("\n\n",len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU\n\n")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(f"\n\nNo GPU\n\n")
    pass

##############

collect_episodes_per_iteration = 1 # @param {type:"integer"} # collect_episode runs this number of episodes per iteration

num_eval_episodes = 100 ## @param {type:"integer"} # this decides how many times compute_avg_return will run. compute_avg_return runs first in the beginning without any training to show random actions. Then collect_episode(...) starts filling the buffer and agent starts training. And in the training process if the current iteration % eval_interval, compute_avg_return(...) runs for num_eval_episodes on eval_env to see append the new rewards. Every time this is run, returns is appended and this is what is shown as the final performance

num_eval_episodes_c51 = 10 ## in the c51 example its 10 so not keeping the usual 1 here

eval_interval = 100 #100 # @param {type:"integer"} # # compute_avg_return called every eval_interval, i.e avg_return calculated filled every eval_interval. this is what is shown in plot 

learning_rate = 1e-3 # @param {type:"number"}

replay_buffer_capacity = 200_000 # @param {type:"integer"} value of max_length, same for both


class UAV_network(py_environment.PyEnvironment):   # network of UAVs not just a single one
    
    def __init__(self, n_users, coverage, name, folder_name, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, MAX_STEPS): # number of total user and the user-coverage for every UAV
        
        '''
        param n_users        : int, no of users.
        param adj_matrix     : matrix as array, adjacency matrix for the network
        param tx_rx_pairs    : list, all tx-rx pairs based on the adjacency matrix
        param tx_users       : list, all users who send data
        
        param RB_total_UL    : int, total RBs available in UL. constant
        param RB_total_DL    : int, total RBs available in DL. constant
        
        param RB_needed_UL   : dict, number of RBs needed to upload data generated by the device at the index. constant
        param RB_needed_DL   : dict, number of RBs needed to download data for the pair at the index. constant
        
        param curr_UL_gen    : dict, generation time of packet currently being uploaded. index user
        param curr_DL_gen    : dict, generation time of packet currently being downloaded. index user_pair
        
        param comp_UL_gen    : dict, generation time of packet that was most recently uploaded successfully. index user
        param comp_DL_gen    : dict, generation time of packet that was most recently downloaded successfully. index user_pair

        param RB_pending_UL  : dict, number of RBs needed to upload pending data generated by the device at the index. changing per slot
        param RB_pending_DL  : dict, number of RBs needed to download pending data generated by the device at the index. changing per slot 
        
        param n_UAVs         : int, number of UAVs in this scenario.
        param coverage       : list of lists, each list has users covered by the drone indicated by the index of the sublist.
        param user_list      : list, user list of all users.
        param UAV_list       : list, generic UAV list. e.g. for 2 UAV [1, 2].
        param users_UAVs     : dict, will contain the users covered by each jth UAV at jth index. user_UAVs[j+1] = self.coverage[j].
        param act_coverage   : dict, same as users_UAVs but 0s removed. actual coverage.
        param BW             : int, bandwidth.
        param UL_capacity    : int, number of users UAV can support in the uplink. previously UAV_capacity
        param DL_capacity    : int, number of users UAV can support in the downlink. previously BS_capacity
        user_locs            : list, locations of the users.
        grid                 : list, grid points where the UAVs can be deployed.
        UAV_loc              : list, UAV deployment positions.
        cover                : list of list, list of list containing users covered by the drone in the index position.
        UAV_age              : dict, age at UAV, i.e. the BS.
        UAV_age_new          : dict, age at UAV including the delay from throughput, i.e. the BS.
        dest_age             : dict, age at the destination nodes. indexed by a tx_rx pair.
        dest_age_new         : dict, age at the destination nodes including the delay from throughput. indexed by a tx_rx pair.
        dest_age_prev        : dict, age at the destination nodes in the previous step. indexed by a tx_rx pair.
        state                : list, state of the system - contains all ages at BS and UAV.
        agent                : Object class, the DL agent that will be shared among all the UAVs.
        actions              : list, set of actions.
        action_size          : int, number of possible actions.
        current_step         : step of the ongoing episode. 1 to MAX_STEP
        episode_step         : int, current step of the ongoing episode. One episode gets over after MAX_STEP number of steps. Note difference with current_step
        preference           : dict, at each index indicated by action is an array and the array has episode wise count of how many times the action was selected. Analogous to visualizing the q-table.
        name                 : string, distinguish between eval and train networks.
        age_dist_UAV         : dict, stores the episode ending age at UAV/BS per user.
        age_dist_dest        : dict, stores the episode ending age at destination nodes per user.
        tx_attempt_dest      : dict, at each index indicated by user is an array and the array has episode wise count of how many times the user was updated.
        tx_attempt_UAV       : dict, at each index indicated by user is an array and the array has episode wise count of how many times the user was sampled.
        
        attempt_sample       : list, index is the episode and the value is the number of times a sample attempt was made. since each sampling results in 1 packet, this value is the number of users selected to sample
        success_sample       : list, index is episode and value is the number of sampling attempts that were successful
        
        attempt_update       : list, index is the episode and the value is the number of times a sample attempt was made. since each sampling results in 1 packet, this value is the number of users selected to sample
        success_update       : list, index is episode and value is the number of sampling attempts that were successful
        sample_time          : dict, stores the slot at which an user was sampled. To show DQN samples at periods
        prev_action_UL       : list, stores the prev action. used for round robin 
        prev_action_DL       : list, stores the prev action. used for round robin 
        dqn_UL_schedule      : dict, schedule that is provided for the power grid simulations
        UL_age_limit         : int, how many times the UL ages were limited to MAX_AGE
        DL_age_limit         : int, how many times the DL ages were limited to MAX_AGE
        BS_location          : 2D arrray, location of the BS
        user_locations       : array of 2D arrays, locations of the users
        MAX_STEPS            : int, number of slots the simulation will run
        packet_lost_upload   : list, each value represents the packets lost in that episode while UL
        packet_lost_download : list, each value represents the packets lost in that episode while DL
        
        attempt_upload  : list, each value represents upload attempts in that episode (lost due to SNR + no RB)
        success_upload  : list, each value represents upload success in that episode
        attempt_download: list, each value represents download attempts in that episode (lost due to SNR + no RB)
        success_download: list, each value represents download success in that episode
                
        packet_lost_upload : list, each value represents packets lost due to SNR in that episode in the UL
        packet_lost_download: list, each value represents packets lost due to SNR in that episode in the DL

        age_dist_UAV_slot_wise : dict, index is slot and value is average age of all users at that slot at the BS
        age_dist_dest_slot_wise: dict, index is slot and value is average age of all users at that slot at dest
        
        best_episodes_average: dict, key is the episode and value is the overall age.
        
        '''
        
        self.n_users        = n_users
        self.adj_matrix     = adj_matrix
        self.tx_rx_pairs    = tx_rx_pairs
        self.tx_users       = tx_users
        
        self.RB_total_UL    = RB_total_UL # constant
        self.RB_total_DL    = RB_total_DL # constant
        
        self.RB_needed_UL   = RB_needed_UL # constant
        self.RB_needed_DL   = RB_needed_DL # constant
        
        self.RB_pending_UL  = {x:0 for x in tx_users} # changing, 0 for for initially when uploading
        self.RB_pending_DL  = {tuple(x):0 for x in tx_rx_pairs} # changing, 0 for initial. will be playing a role only after a packet has been completely uploaded
        
        self.curr_UL_gen    = {x:-1 for x in tx_users} # # will become a valid time only after a packet has completely been UL
        self.curr_DL_gen    = {tuple(x):-1 for x in tx_rx_pairs} # will become a valid time only after a packet has started DL
        
        self.comp_UL_gen    = {x:-1 for x in tx_users}
        self.comp_DL_gen    = {tuple(x):-1 for x in tx_rx_pairs}

        self.periodicity    = periodicity
        self.coverage       = coverage
        self.n_UAVs         = len(coverage)

        self.users_UAVs     = {} # [i for i in range(1, self.n_users + 1)] # updated in start_network()
        self.act_coverage   = {} # updated in start_network()
        self.user_locs      = []
        self.grid           = []
        self.UAV_loc        = []
        self.cover          = []
        self.actions_space  = [] # initialized once the coverage is calculated
        self.action_size    = 2 # will be updated in start_network()
        self.episode_step   = 0
        self.preference     = {}
        self.current_step   = 1
        self.UAV_age        = {}
        self.UAV_age_new    = {}
        self.dest_age       = {} ## previously BS age
        self.dest_age_prev  = {} ## previously BS_age_prev
        self.name           = name
        self.age_dist_UAV   = {}
        self.age_dist_dest  = {}    
        self.tx_attempt_dest= {}
        self.tx_attempt_UAV = {}
        self.user_list      = []
        self.UAV_list       = []
        self.folder_name    = folder_name
        self.packet_upload_loss_thresh    = packet_upload_loss_thresh
        self.packet_download_loss_thresh  = packet_download_loss_thresh

        self.sample_time    = {} ## goes from 1 to MAX_STEPS inclusive in all cases
        self.prev_action_UL = None
        self.prev_action_DL = None
        self.dqn_UL_schedule={} ##    key:([slot 1st UL completed, gen time of the packet], [slot 2nd UL completed, gen time of the packet], [...], [...])
        self.dqn_DL_schedule={} ##    key:([slot 1st DL completed, gen time of the packet], [slot 2nd DL completed, gen time of the packet], [...], [...])
        self.UL_age_limit   = 0
        self.DL_age_limit   = 0
        self.BS_location    = BS_location
        self.user_locations = user_locations
        self.MAX_STEPS      = MAX_STEPS

        
        self.attempt_upload = []
        self.success_upload = []
        self.attempt_download = []
        self.success_download = []
        
        self.packet_lost_upload = []
        self.packet_lost_download = []
        
        self.PDR_upload = []
        self.PDR_download = [] 

        self.age_dist_UAV_slot_wise = {}
        self.age_dist_dest_slot_wise = {}
        

        self.best_episodes_average = {}
        self.best_episodes_peak = {}
        
        if verbose:
            print(f"\n\ntx_rx_pairs = {self.tx_rx_pairs} with length {len(self.tx_rx_pairs)} and tx_users = {self.tx_users} with length {len(self.tx_users)}")
            print(f"tx_rx_pairs = {self.tx_rx_pairs} with length {len(self.tx_rx_pairs)} and tx_users = {self.tx_users} with length {len(self.tx_users)}", file = open(self.folder_name + "/results.txt", "a"))
        ## relevant pair and tx users calculation ends
        
        # assert(len(self.tx_users)>=self.RB_total_UL and len(self.tx_rx_pairs)>=self.RB_total_DL) ## as the action space is based on permutation where user is top and RB is bottom
        
        for ii in range(1, MAX_STEPS+1):
            self.age_dist_UAV_slot_wise[ii] = []
            self.age_dist_dest_slot_wise[ii] = []
         
        self.start_network()

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=self.action_size-1, name='sample_update')  # bounds are inclusive
        
        if CSI_as_state:
            if sample_error_in_CSI:
                self._observation_spec = array_spec.BoundedArraySpec(shape=(1 , len(list([self.current_step])) + 3*n_users + len(self.UAV_list)), dtype=np.float32, minimum=0, maximum=MAX_AGE, name='current_state')
                self._state = np.concatenate((list([self.current_step]), [1]*2*(n_users), list(self.packet_download_loss_thresh.values()), list(self.packet_upload_loss_thresh.values())), axis=None) #
            else:
                self._observation_spec = array_spec.BoundedArraySpec(shape=(1 , len(list([self.current_step])) + 2*n_users + len(self.UAV_list)), dtype=np.float32, minimum=0, maximum=MAX_AGE, name='current_state')
                self._state = np.concatenate((list([self.current_step]), [1]*2*(n_users), list(self.packet_upload_loss_thresh.values())), axis=None) #
            
        if time_in_state == False: ## the state doesn't have t
            self._observation_spec = array_spec.BoundedArraySpec(shape=(1 , n_users + len(tx_rx_pairs)), dtype=np.float32, minimum=0, maximum=MAX_AGE, name='current_state')
            self._state = np.concatenate(( [0]*(n_users + len(tx_rx_pairs))), axis=None)
            
        else:       ## time_in_state == true      
            self._observation_spec = array_spec.BoundedArraySpec(shape=(1 , len(list([self.current_step])) + n_users + len(tx_rx_pairs)), dtype=np.float32, minimum=0, maximum=MAX_AGE, name='current_state')
            self._state = np.concatenate((list([self.current_step]), [0]*(n_users + len(tx_rx_pairs))), axis=None) #
            

        # if verbose:
        # print(f"initial state is {self._state} with length {np.shape(self._state)} when CSI_as_state = {CSI_as_state} and sample_error_in_CSI = {sample_error_in_CSI}")  
              
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    
    def update_act_coverage(self):
        ## remove the padding from the coverage 0s so that we get the actual coverage
    
        n = [i for i in list(self.users_UAVs.keys())] # UAV ids
        k = [i for i in list(self.users_UAVs.values())] # all users
        b = {} # this will be the act_coverage

        for i in range(len(self.coverage)): # for each UAV ## don't make it UAV_list as it has not been made yet
            old_list = self.users_UAVs[i] # users under UAV i
            new_list = [j for j in old_list if j!=0] # 0s are padding not actual user. user ID cannot be 0, drone ID can be 0
            b[i] = new_list

        return b
    
    def start_network(self):
        
        for i in range(len(self.coverage)): ## don't make it UAV_list as it has not been made yet
            self.users_UAVs[i] = self.coverage[i]
            # index of users_UAVs will start from 0
            
        self.act_coverage   = self.update_act_coverage()   
        self.user_list      = functools.reduce(operator.iconcat, list(self.act_coverage.values()), []) # list(self.act_coverage.values())
        self.UAV_list       = list(self.act_coverage.keys())
        
        if verbose:
            print(f"user list = {self.user_list}")
            print(f"UAV list  = {self.UAV_list}")
        
        for ii in self.user_list:
            self.sample_time[ii] = []


        if verbose:
            print(f'self.n_users = {self.n_users}, self.n_UAVs = {self.n_UAVs}, self.act_coverage = {self.act_coverage}, self.update_loss_thresh = {self.packet_upload_loss_thresh}, self.sample_loss_thresh = {self.packet_download_loss_thresh}, self.UAV_list = {self.UAV_list}, self.user_list = {self.user_list}')
            # time.sleep(15)

        # self.create_action_space() # @ remove_action biplav

        # not doing in initialize_age() as initialize_age() is run every time net is reset so older values will be lost. start_network is run only once
        for i in self.user_list:
            self.age_dist_UAV[i]   = [] # ending age of every episode, updated in _reset
            self.tx_attempt_UAV[i] = [] 
            
        for i in self.tx_rx_pairs:
            self.age_dist_dest[tuple(i)]  = []
            self.tx_attempt_dest[tuple(i)]  = [] # number of attempts per episode, initialize in _reset and updated in _step  
            
        for i in range(self.action_size):
            self.preference[i] = []
            
        for i in self.tx_users:
            self.dqn_UL_schedule[i] = [] ## list corresponding to each key

        for i in self.tx_rx_pairs:
            self.dqn_DL_schedule[tuple(i)] = [] ## list corresponding to each key
            
        if verbose:
            print(f'\n{self.name} started and age_dist_UAV = {self.age_dist_UAV}, age_dist_dest = {self.age_dist_dest}, tx_attempt_dest = {self.tx_attempt_dest}, tx_attempt_UAV = {self.tx_attempt_UAV}\n')

    
    def initialize_age(self):    
       
        # before initializing, save the info    
        # for the first time it is run, BS_age and others haven't even been initialized  
        if self.episode_step!=1: ## why !=1 ??, see above point
            
            if len(self.success_upload)>0 and self.success_upload[-1]!=0 and self.attempt_upload[-1] !=0 and self.success_download[-1] !=0 and self.attempt_download[-1] !=0:
                
                if verbose:
                    
                    print(f"attempt_download={self.attempt_download}, attempt_upload={self.attempt_upload}, success_download = {self.success_download}, attempt_download={self.attempt_download}")
            
                self.PDR_upload.append(self.success_upload[-1]/self.attempt_upload[-1])
                self.PDR_download.append(self.success_download[-1]/self.attempt_download[-1])
                        
            self.attempt_upload.append(0)
            self.success_upload.append(0)
            
            self.attempt_download.append(0)
            self.success_download.append(0)
            
            
            ## -2 as -1 will have the 0 we add in the previous lines
            
            self.packet_lost_download.append(0)
            self.packet_lost_upload.append(0)

            
            for i in self.user_list:
                self.age_dist_UAV[i].append(self.UAV_age[i])
                self.tx_attempt_UAV[i].append(0)
                
            for i in self.tx_rx_pairs:
                self.age_dist_dest[tuple(i)].append(self.dest_age[tuple(i)])
                self.tx_attempt_dest[tuple(i)].append(0) ## was tx_attempt_BS

   
            for i in range(self.action_size):
                self.preference[i].append(0) 
                
            self.age_dist_dest_slot_wise[self.current_step].append(np.mean(list(self.dest_age.values())))
            self.age_dist_UAV_slot_wise[self.current_step].append(np.mean(list(self.UAV_age.values())))
                
            
            if verbose:
                print(f'\n{self.name} just before reset of {self.name} the age at UAV = {self.UAV_age}, age at dest = {self.dest_age}')  # and these have used to update age_dist_UAV = {self.age_dist_UAV} and age_dist_dest = {self.age_dist_dest}\n')
                # print(f'\n{self.name} in the same reset block, tx_attempts have been updated as tx_attempt_UAV = {self.tx_attempt_UAV} and tx_attempt_dest = {self.tx_attempt_dest}\n')
                # print(f"{self.name} preference is {self.preference}")
        
        for i in self.user_list:
            # initial age put 1 and not 0 as if 0, in first time step whethere sampled or not, all users age at UAV becomes 1 but for 1, it is different - 2 for not sampled and 1 for sampled
            self.UAV_age[i] = 0

        for i in self.tx_rx_pairs:
            self.dest_age[tuple(i)]  = 0
            self.dest_age_prev[tuple(i)] = 0 # special case for first step ??

        self.RB_pending_UL  = {x:0 for x in self.tx_users} 
        self.RB_pending_DL  = {tuple(x):0 for x in self.tx_rx_pairs}
        
        self.curr_UL_gen    = {x:-1 for x in self.tx_users} 
        self.curr_DL_gen    = {tuple(x):-1 for x in self.tx_rx_pairs}
        
        self.comp_UL_gen    = {x:-1 for x in self.tx_users}
        self.comp_DL_gen    = {tuple(x):-1 for x in self.tx_rx_pairs}


    def _reset(self):
        self.episode_step +=1
            
        if CSI_as_state: # csi as state needed
            if sample_error_in_CSI: # sampling error as state needed  
                self._state = np.concatenate((list([self.current_step]), [0]*2*(len(self.user_list)), list(self.packet_download_loss_thresh.values()), list(self.packet_upload_loss_thresh.values())), axis=None)
            else:
                self._state = np.concatenate((list([self.current_step]), [0]*2*(len(self.user_list)), list(self.packet_upload_loss_thresh.values())), axis=None)

            
        if time_in_state == False:
        #    self._observation_spec = array_spec.BoundedArraySpec(shape=(1 , self.n_users + len(self.tx_rx_pairs)), dtype=np.float32, minimum=0, maximum=len(self.tx_rx_pairs)+2, name='current_state')
           self._state = np.concatenate(( [0]*(self.n_users + len(self.tx_rx_pairs))), axis=None)
           
        else:
            self._state = np.concatenate((list([self.current_step]), [0]*(self.n_users + len(self.tx_rx_pairs))), axis=None) #
   
        self._episode_ended = False
        self.current_step = 1
        if verbose:
            print(f'\n{self.name} after reset, episode {self.episode_step} begins with self._state = {self._state} with shape {np.shape(self._state)} when CSI_as_state = {CSI_as_state}, sample_error_in_CSI = {sample_error_in_CSI}, time_in_state = {time_in_state}\n') 
            # time.sleep(10)
         # just before initializing age, this episode ending age to be saved   
        self.initialize_age()
        return ts.restart(np.array([self._state], dtype=np.float32))

        
    def map_actions(self, action):  
        '''
        convert the single integer action to specific sampling and updating tasks
        '''
        # print(f'inside  map_actions, action={action}, type(action)={type(action)}')
        # print(f'action={action}, self.actions_space={self.actions_space}')
        actual_action = self.actions_space[action]
        if verbose:
            # print(f'action space is {self.actions_space}, length is {self.action_size}, array size is {len(self.actions_space)} selected action is {action} which maps to {actual_action}')
            pass
        return actual_action
    
    def get_current_state(self): #  
        # doesn't change anything, just returns the current state. Ages have been updated in the take_RL_action, here the new state is returned
        state_UAV = np.array(list(self.UAV_age.values()))
        state_dest  = np.array(list(self.dest_age.values()))
        
        if CSI_as_state:
            if sample_error_in_CSI:
                self._state = np.concatenate((list([self.current_step]), state_UAV, state_dest, list(self.packet_download_loss_thresh.values()) , list(self.packet_upload_loss_thresh.values())), axis=None) 
            else:
                self._state = np.concatenate((list([self.current_step]), state_UAV, state_dest, list(self.packet_upload_loss_thresh.values())), axis=None) 
                # newlist = [x for x in fruits if "a" in x]

        else:
            if self.current_step>MAX_AGE:
                current_step_with_limit = MAX_AGE
            else:
                current_step_with_limit = self.current_step
            state_list = np.concatenate((list([current_step_with_limit]), state_UAV, state_dest), axis=None)
            # print(f"state_list = {state_list}")
            state_list = [MAX_AGE if x>MAX_AGE else x for x in state_list]
            # print(f"state_list = {state_list}")
            self._state = state_list
            
        if time_in_state == False: # state_UAV, state_dest already have been capped as the ages are capped
            self._state = np.concatenate(( state_UAV, state_dest), axis=None) 

        if verbose:
            print(f'\nself._state of {self.name} get_current_state() = {self._state} with shape = {np.shape(self._state)}\n') # debug
        return (self._state)
    
    def create_action_space(self):
        '''
        for 1 UAV once the coverage has been decided, create the action space
        sample means sender to UAV, update means UAV to receiver
        '''
        
        # print(f"inside create_action_space")
        ## update start # UAV to dest nodes
        # download_pair_possibilities = list(itertools.permutations(self.tx_rx_pairs, len(self.tx_rx_pairs)))
        download_pair_possibilities = list(itertools.permutations(self.tx_rx_pairs, self.RB_total_DL)) # as user will not exceed total RB
        ## update action part done
        
         
         
        # sample start
        # upload_user_possibilities = list(itertools.permutations(self.tx_users, len(self.tx_users)))
        upload_user_possibilities = list(itertools.permutations(self.tx_users, self.RB_total_UL))

        # sample action part done

        # print(f"update_user_possibilities = {update_user_possibilities}, {type(update_user_possibilities)}")
        # print(f"all_user_sampling_combinations = {all_user_sampling_combinations}, {type(all_user_sampling_combinations)}")
        
        # time.sleep(2)
        

        actions_space = list(itertools.product(upload_user_possibilities, download_pair_possibilities))
        
        actions_space = [list(i) for i in actions_space]
            
        # if verbose:
        #     print(f"tx = {sample_user_possibilities} with length = {len(sample_user_possibilities)} and \nall_user_sampling_combinations is {all_user_sampling_combinations} with length {len(all_user_sampling_combinations)}")
        #     print("\naction_size is ", len(actions_space)) #, " and they are actions_space = ", actions_space)
        #     # time.sleep(10)
            
            
        self.actions_space = actions_space
        self.action_size = len(self.actions_space)
        
        if verbose:
            print(f"\n{self.name} has a action_space of size ", np.shape(self.actions_space)) #, " and they are ", self.actions_space,  "\n")
            
        # print("\n action_space is of size ", np.shape(self.actions_space), file = open(self.folder_name + "/results.txt", "a"))
        # print("\n action_space is of size ", np.shape(self.actions_space), " and they are ", self.actions_space)
    
    def _step(self, action):
        # print("step ", self.current_step, " started") ## runs for MAX_STEPS steps
        # each step returns TimeStep(step_type, reward, discount, observation

                  
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            if verbose:
                print(f'for {self.name}, episode = {self.episode_step} at first reset')
            return self.reset()
        
        actual_action = self.map_actions(action)
        action = action.tolist()
        
        # print(f"\n env = {self.name}, self.current_step = {self.current_step}, self.episode_step = {self.episode_step}, action = {action}, type(action) = {type(action)}, (actual_action)={actual_action}, preference = {self.preference}") #, {type(self.preference[action])}") ## for some reason action is type nparray
        # self.preference[action][-1] = self.preference[action][-1] + 1
            
        download_user_pairs = list(actual_action[1])
        upload_users   = list(actual_action[0])
        
        if verbose:
        
            print(f'\ncurrent_step = {self.current_step}, selected action = {action}, actual_dqn_action={actual_action}, upload_users = {upload_users} sampled_users={download_user_pairs}\n') 
            # time.sleep(3)
            
            # print(f"{self.name} tx_attempt_dest was {self.tx_attempt_dest}")
            
            ## download starts     
        
        remaining_RB_DL = self.RB_total_DL
        
        for i in download_user_pairs:

            self.attempt_download[-1] = self.attempt_download[-1] + 1
            
            if verbose:
                print(f"\ncurrent slot = {self.current_step}, pair {i} age at the beginning is {self.dest_age[tuple(i)]}")

            received_SNR_download = getSNR(self.BS_location, self.user_locations[i[1]])
            
            if received_SNR_download < self.packet_download_loss_thresh[tuple(i)]:
                self.packet_lost_download[-1] = self.packet_lost_download[-1] + 1   

            if (remaining_RB_DL) > 0 and (received_SNR_download > self.packet_download_loss_thresh[tuple(i)]): # scheduling only when RB available and reception energy exceeds threshold
                
                self.success_download[-1] = self.success_download[-1] + 1
                
                if self.comp_UL_gen[i[0]] == -1: # means no packet has been uploaded yet by the source
                    if verbose:
                        print(f'pair {i}, i.e. user {i[0]} has no data yet at the BS so empty packet will be sent')
                    
                # if (self.comp_DL_gen[tuple(i)]!=self.comp_UL_gen[i[0]]): # and (self.comp_DL_gen[tuple(i)]!=self.curr_DL_gen[tuple(i)]):  ## ?? which one needed ?
                    
                # before DL a new packet, check if it has already been DL. Might happen when no new uploaded but DL keeps getting scheduled
                # true means the packet currently to be DL (comp_UL_gen[i[0]]) is different than the prev DL (comp_DL_gen[tuple(i)])
                

                assert (self.comp_DL_gen[tuple(i)] <= self.comp_UL_gen[i[0]]) # as BS will have newer or same packet that has been recently DL to dest completely.
                        
                if self.RB_pending_DL[tuple(i)]==0: # prev packet fully downloaded in prev attempt, so download the recently uploaded packet    
                    if verbose:
                        print(f"pair {i} completed DL in prev attempt so new packet DL. old values-curr_DL_gen[{i}]={self.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={self.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={self.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={self.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={self.RB_pending_DL[tuple(i)]}, remaining_RB_DL={remaining_RB_DL}")       
                                                            
                    
                    self.curr_DL_gen[tuple(i)]   = self.comp_UL_gen[i[0]] # already uploaded packet
                    self.RB_pending_DL[tuple(i)] = self.RB_needed_DL[tuple(i)] # as new packet
                    assigned_RB_DL  = min(remaining_RB_DL, self.RB_pending_DL[tuple(i)])
                    remaining_RB_DL = remaining_RB_DL - assigned_RB_DL
                    self.RB_pending_DL[tuple(i)] = self.RB_pending_DL[tuple(i)] - assigned_RB_DL # new pending after assignment
                    # new packet details ended

                    assert(assigned_RB_DL>-1 and self.RB_pending_DL[tuple(i)]>-1)
                    
                    if (self.comp_DL_gen[tuple(i)]==self.curr_DL_gen[tuple(i)]) and (self.comp_DL_gen[tuple(i)]!=-1): # comp_DL_gen is from the prev completed attempt and curr_DL_gen is newly set, so comparision correct. will only happen when DL a new packet
                        if verbose:
                            print(f"packet being DL has been already DL")  
                                            
                    if self.RB_pending_DL[tuple(i)] == 0: # means packet was fully downloaded in this slot
                        # finish off DL of new packet
                        # self.dest_age[tuple(i)] = self.current_step
                        

                        ## MOD
                        if self.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                            #     self.dest_age[tuple(i)] = self.current_step
                            if self.current_step<=MAX_AGE:
                                self.dest_age[tuple(i)] = self.current_step
                            else:
                                self.dest_age[tuple(i)] = MAX_AGE
                                self.DL_age_limit = self.DL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")


                        else: ## current packet started DL after BS had a packet
                            # self.dest_age[tuple(i)] = self.current_step - self.curr_DL_gen[tuple(i)] # age change after packet fully sent
                            
                            ## MOD
                            if (self.current_step - self.curr_DL_gen[tuple(i)])<=MAX_AGE:
                                self.dest_age[tuple(i)] = self.current_step - self.curr_DL_gen[tuple(i)]
                            else:
                                self.dest_age[tuple(i)] = MAX_AGE
                                self.DL_age_limit = self.DL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                            
                            # record schedule only if (i) valid packet downloaded (ii) packet fully downloaded
                            if (random_episodes-self.episode_step)<100: ##means the last 100 episodes
                                self.dqn_DL_schedule[tuple(i)].append([self.episode_step, self.current_step, self.curr_DL_gen[tuple(i)]])
                           
                        self.comp_DL_gen[tuple(i)] = self.curr_DL_gen[tuple(i)]
                        
                        # older packet done
                        if verbose:
                            print(f"pair {i} age at the end is {self.dest_age[tuple(i)]}. completed DL new pack in same slot. new values-curr_DL_gen[{i}]={self.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={self.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={self.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={self.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={self.RB_pending_DL[tuple(i)]},assigned_RB_DL = {assigned_RB_DL}, remaining_RB_DL={remaining_RB_DL}\n")
                        
                    else: # self.RB_pending_DL[tuple(i)] != 0. new packet incomplete DL
                            
                        if self.comp_DL_gen[tuple(i)] == -1: ## no packet DL till now 
                            # self.dest_age[tuple(i)] = self.current_step  
                            
                            ## MOD
                            if (self.current_step)<=MAX_AGE:
                                self.dest_age[tuple(i)] = self.current_step
                            else:
                                self.dest_age[tuple(i)] = MAX_AGE
                                self.DL_age_limit = self.DL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                            
                                                        
                        else: ## at least a packet had been downloaded
                            # self.dest_age[tuple(i)] = self.dest_age[tuple(i)] + 1 # age change after packet partial sent
                            
                            ## MOD
                            if (self.dest_age[tuple(i)] + 1)<=MAX_AGE:
                                self.dest_age[tuple(i)] = self.dest_age[tuple(i)] + 1
                            else:
                                self.dest_age[tuple(i)] = MAX_AGE
                                self.DL_age_limit = self.DL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                            
                        if verbose:
                            print(f"pair {i} age at the end is {self.dest_age[tuple(i)]}. partial DL new pack in current attempt. new values-curr_DL_gen[{i}]={self.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={self.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={self.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={self.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={self.RB_pending_DL[tuple(i)]},assigned_RB_DL = {assigned_RB_DL}, remaining_RB_DL={remaining_RB_DL}\n") 
                    
                else: ## self.RB_pending_DL[tuple(i)]!=0
                # old packet from prev attempts is still being downloaded. part of older packet and part of newer packet will not be done together even if RBs available. Attempt to transmit the older packet fully here
                    
                    if verbose:
                        print(f"pair {i} continue incomplete DL from prev. old values-curr_DL_gen[{i}]={self.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={self.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={self.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={self.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={self.RB_pending_DL[tuple(i)]}, remaining_RB_DL={remaining_RB_DL}")
                    
                    assigned_RB_DL  = min(remaining_RB_DL, self.RB_pending_DL[tuple(i)])
                    remaining_RB_DL = remaining_RB_DL - assigned_RB_DL
                    self.RB_pending_DL[tuple(i)] = self.RB_pending_DL[tuple(i)] - assigned_RB_DL
                    
                    
                    if self.RB_pending_DL[tuple(i)] == 0: # means old packet was fully downloaded in this slot
                        # finish off DL of selected packet
                        if self.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                            # self.dest_age[tuple(i)] = self.current_step
                            
                            ## MOD
                            if (self.current_step)<=MAX_AGE:
                                self.dest_age[tuple(i)] = self.current_step
                            else:
                                self.dest_age[tuple(i)] = MAX_AGE
                                self.DL_age_limit = self.DL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                            
                            
                        else: ## current packet started DL after BS had a packet
                            # self.dest_age[tuple(i)] = self.current_step - self.curr_DL_gen[tuple(i)] # age change after packet fully sent
                            
                            ## MOD
                            if (self.current_step - self.curr_DL_gen[tuple(i)])<=MAX_AGE:
                                self.dest_age[tuple(i)] = self.current_step - self.curr_DL_gen[tuple(i)]
                            else:
                                self.dest_age[tuple(i)] = MAX_AGE
                                self.DL_age_limit = self.DL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                            
                            self.comp_DL_gen[tuple(i)] = self.curr_DL_gen[tuple(i)]
                        # older packet done
                        
                            # record schedule only if (i) valid packet downloaded (ii) packet fully downloaded
                            if (random_episodes-self.episode_step)<100: ##means the last 100 episodes
                                self.dqn_DL_schedule[tuple(i)].append([self.episode_step, self.current_step, self.curr_DL_gen[tuple(i)]])

                        if verbose:
                            print(f"pair {i} age at the end is {self.dest_age[tuple(i)]}. old packet fully DL. new values-curr_DL_gen[{i}]={self.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={self.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={self.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={self.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={self.RB_pending_DL[tuple(i)]},assigned_RB_DL = {assigned_RB_DL}, remaining_RB_DL={remaining_RB_DL}\n")
                            
                    else: # self.RB_pending_DL[tuple(i)] != 0: # means old packet wasn't fully downloaded in this slot
                        if self.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                            # self.dest_age[tuple(i)] = self.current_step
                            
                            ## MOD
                            if (self.current_step)<=MAX_AGE:
                                self.dest_age[tuple(i)] = self.current_step
                            else:
                                self.dest_age[tuple(i)] = MAX_AGE
                                self.DL_age_limit = self.DL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                            
                        else: ## current packet started DL after BS had a packet
                            # self.dest_age[tuple(i)] = self.dest_age[tuple(i)] + 1 # age change after packet fully sent
                            
                            
                            ## MOD
                            if (self.dest_age[tuple(i)] + 1)<=MAX_AGE:
                                self.dest_age[tuple(i)] = self.dest_age[tuple(i)] + 1
                            else:
                                self.dest_age[tuple(i)] = MAX_AGE
                                self.DL_age_limit = self.DL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                            
                            if verbose:
                                print(f"pair {i} age at the end is {self.dest_age[tuple(i)]}. old packet again partial DL. new values-curr_DL_gen[{i}]={self.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={self.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={self.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={self.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={self.RB_pending_DL[tuple(i)]},assigned_RB_DL = {assigned_RB_DL}, remaining_RB_DL={remaining_RB_DL}\n")  
                    
                    assert(assigned_RB_DL>-1 and remaining_RB_DL>-1 and self.RB_pending_DL[tuple(i)]>-1)
                        

            else: # no RB remaining
                if self.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                    # self.dest_age[tuple(i)] = self.current_step
                    
                    ## MOD
                    if (self.current_step)<=MAX_AGE:
                        self.dest_age[tuple(i)] = self.current_step
                    else:
                        self.dest_age[tuple(i)] = MAX_AGE
                        self.DL_age_limit = self.DL_age_limit + 1
                        if verbose:
                            print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                        
                    
                else: ## current packet started DL after BS had a packet
                    # self.dest_age[tuple(i)] = self.dest_age[tuple(i)] + 1 # age change after packet fully sent    
                    
                    ## MOD
                    if (self.dest_age[tuple(i)] + 1)<=MAX_AGE:
                        self.dest_age[tuple(i)] = self.dest_age[tuple(i)] + 1
                    else:
                        self.dest_age[tuple(i)] = MAX_AGE
                        self.DL_age_limit = self.DL_age_limit + 1
                        if verbose:
                            print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                    
                                    
                if verbose:
                    print(f"remaining_RB_DL = {remaining_RB_DL}, so pair {i} not scheduled. pair {i} age at the end is {self.dest_age[tuple(i)]}\n")

        for i in self.tx_rx_pairs: 
            if i not in download_user_pairs:
                if verbose:                
                    print(f"\npair {i} age at the beginning is {self.dest_age[tuple(i)]}")
                if self.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                    # self.dest_age[tuple(i)] = self.current_step
                    
                    ## MOD
                    if (self.current_step)<=MAX_AGE:
                        self.dest_age[tuple(i)] = self.current_step
                    else:
                        self.dest_age[tuple(i)] = MAX_AGE
                        self.DL_age_limit = self.DL_age_limit + 1
                        if verbose:
                            print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")
                    
                                        
                else: ## current packet started DL after BS had a packet
                    # self.dest_age[tuple(i)] = self.dest_age[tuple(i)] + 1 # age change after packet fully sent       
                    
                    ## MOD
                    if (self.dest_age[tuple(i)] + 1)<=MAX_AGE:
                        self.dest_age[tuple(i)] = self.dest_age[tuple(i)] + 1
                    else:
                        self.dest_age[tuple(i)] = MAX_AGE   
                        self.DL_age_limit = self.DL_age_limit + 1
                        if verbose:
                            print(f"MAX_AGE for DL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}")            
                
                if verbose:
                    print(f"download_user_pairs={download_user_pairs}, so pair {i} not selected. pair {i} age at the end is {self.dest_age[tuple(i)]}\n")
    
        ## download ends     
            
        ## uploading process start
            
        remaining_RB_UL = self.RB_total_UL ## all RBs available in the beginning of the slot
                    
        for i in upload_users:
            if verbose:
                print(f"\nuser {i} age at the beginning is {self.UAV_age[i]} with period {self.periodicity[i]}")
                
            self.attempt_upload[-1] = self.attempt_upload[-1] + 1
            
            received_SNR_upload = getSNR(self.BS_location, self.user_locations[i])
            
            if received_SNR_upload < self.packet_upload_loss_thresh[i]:
                self.packet_lost_upload[-1] = self.packet_lost_upload[-1] + 1

            
            if (remaining_RB_UL > 0) and (received_SNR_upload > self.packet_upload_loss_thresh[i]): ## only then scheduling can be done
                
                self.success_upload[-1] = self.success_upload[-1] + 1
                                        
                if self.RB_pending_UL[i]==0: ## this user has completed the prev packet in its prev attempt, switch to a new packet now. both sample and try to send the new packet. also for the first slot, this wil be true  
                    
                    if verbose:
                        print(f"user {i} completed UL in its prev attempt. old values-curr_UL_gen[{i}] = {self.curr_UL_gen[i]}, comp_UL_gen[{i}] = {self.comp_UL_gen[i]}, RB_pending_UL[{i}] = {self.RB_pending_UL[i]}, remaining_RB_UL = {remaining_RB_UL}.")
                    
                    # new packet details started
                    if self.current_step%self.periodicity[i]==0: # generate at will has period 1 so %=0
                        last_pack_generated = self.current_step  # generation time of the packet sampled now
                        # print(f"last_pack_generated 1 = {last_pack_generated}. self.current_step%self.periodicity[i]={self.current_step%self.periodicity[i]}")
                    else:
                        last_pack_generated = self.periodicity[i]*max(math.floor(self.current_step/self.periodicity[i]), 1) ## max added so that the result is never 0 as our slots start from 1. so periodicity of 2 means 2,4,6,8
                        # print(f"last_pack_generated 2 = {last_pack_generated}. self.current_step%self.periodicity[i]={self.current_step%self.periodicity[i]}")

                    
                    if self.current_step >= last_pack_generated: # else will stay -1
                        self.curr_UL_gen[i]  = last_pack_generated # self.current_step # sample at will, so new packet generated and sampled only if current time and last generated time is feasible
                    
                    if verbose:
                        print(f"last_pack_generated = {last_pack_generated}")                    
                                                
                    self.RB_pending_UL[i] = self.RB_needed_UL[i] # as new packet
                    assigned_RB_UL  = min(remaining_RB_UL, self.RB_pending_UL[i])
                    remaining_RB_UL = remaining_RB_UL - assigned_RB_UL
                    self.RB_pending_UL[i] = self.RB_pending_UL[i] - assigned_RB_UL # new pending after assignment
                    assert(assigned_RB_UL>-1 and remaining_RB_UL>-1 and self.RB_pending_UL[i]>-1)
                    # new packet details ended
                    
                    if self.RB_pending_UL[i] == 0: # means packet was fully uploaded in this slot
                        # finish off details of completed packet
                        if self.curr_UL_gen[i] == -1: ## current packet started UL when device had nothing 
                            # self.UAV_age[i] = self.current_step
                            
                            ## MOD
                            if (self.current_step)<=MAX_AGE:
                                self.UAV_age[i] = self.current_step
                            else:
                                self.UAV_age[i] = MAX_AGE 
                                self.UL_age_limit = self.UL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for UL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}") 
                            
                        else: ## current packet started UL after device had a packet
                            # self.UAV_age[i] = self.current_step - self.curr_UL_gen[i] # age change after packet fully sent      
                            
                            ## MOD
                            if (self.current_step - self.curr_UL_gen[i])<=MAX_AGE:
                                self.UAV_age[i] = self.current_step - self.curr_UL_gen[i]
                            else:
                                self.UAV_age[i] = MAX_AGE 
                                self.UL_age_limit = self.UL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for UL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}") 
                            
                                              
                            self.comp_UL_gen[i] = self.curr_UL_gen[i] # no need of multiple cases like DL
                            
                            # record schedule only if (i) valid packet downloaded (ii) packet fully downloaded
                            if (random_episodes-self.episode_step)<100: ##means the last 100 episodes
                                self.dqn_UL_schedule[i].append([self.episode_step, self.current_step, self.curr_UL_gen[i]])
                        
                        # new packet done
                        
                        if verbose:
                            print(f"user {i} age at the end is {self.UAV_age[i]}. new packet fully UL in same slot-new values curr_UL_gen[{i}] = {self.curr_UL_gen[i]}, comp_UL_gen[{i}] = {self.comp_UL_gen[i]}, RB_pending_UL[{i}] = {self.RB_pending_UL[i]}, assigned_RB_UL = {assigned_RB_UL}, remaining_RB_UL = {remaining_RB_UL}\n")
                    
                    else: # means packet was not fully uploaded in this slot
                        # self.UAV_age[i] = self.UAV_age[i] + 1
                        
                        ## MOD
                        if (self.UAV_age[i] + 1)<=MAX_AGE:
                            self.UAV_age[i] = self.UAV_age[i] + 1
                        else:
                            self.UAV_age[i] = MAX_AGE 
                            self.UL_age_limit = self.UL_age_limit + 1
                            if verbose:
                                print(f"MAX_AGE for UL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}") 

                        if verbose:
                            print(f"user {i} age at the end is {self.UAV_age[i]}. new packet partial UL-new values curr_UL_gen[{i}] = {self.curr_UL_gen[i]}, comp_UL_gen[{i}] = {self.comp_UL_gen[i]}, RB_pending_UL[{i}] = {self.RB_pending_UL[i]}, assigned_RB_UL = {assigned_RB_UL}, remaining_RB_UL = {remaining_RB_UL}\n")
                
                else: # old packet from prev attempts is still being sent. part of older packet and part of newer packet will not be done together even if RBs available. Attempt to transmit the older packet fully
                    
                    if verbose:
                        print(f"continue incomplete UL from prev-old values-curr_UL_gen[{i}] = {self.curr_UL_gen[i]}, comp_UL_gen[{i}] = {self.comp_UL_gen[i]}, RB_pending_UL[{i}] = {self.RB_pending_UL[i]}, remaining_RB_UL = {remaining_RB_UL}")
                    
                    assigned_RB_UL = min(remaining_RB_UL, self.RB_pending_UL[i])
                    remaining_RB_UL = remaining_RB_UL - assigned_RB_UL
                    self.RB_pending_UL[i] = self.RB_pending_UL[i] - assigned_RB_UL
                    assert(assigned_RB_UL>-1 and remaining_RB_UL>-1 and self.RB_pending_UL[i]>-1)

                    if self.RB_pending_UL[i] == 0: # means the partial packet was fully uploaded in this slot
                        # finish off details of completed packet
                        if self.curr_UL_gen[i] == -1: ## current packet started UL when device had nothing 
                            # self.UAV_age[i] = self.current_step
                            
                           ## MOD
                            if (self.current_step)<=MAX_AGE:
                                self.UAV_age[i] = self.current_step
                            else:
                                self.UAV_age[i] = MAX_AGE 
                                self.UL_age_limit = self.UL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for UL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}") 
                            
                        else: ## current packet started UL after device had a packet
                            # self.UAV_age[i] = self.current_step - self.curr_UL_gen[i] # age change after packet fully sent       
                            
                            ## MOD
                            if (self.current_step - self.curr_UL_gen[i])<=MAX_AGE:
                                self.UAV_age[i] = self.current_step - self.curr_UL_gen[i]
                            else:
                                self.UAV_age[i] = MAX_AGE 
                                self.UL_age_limit = self.UL_age_limit + 1
                                if verbose:
                                    print(f"MAX_AGE for UL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}") 
                                             
                            self.comp_UL_gen[i] = self.curr_UL_gen[i] # no need of multiple cases like DL
                            
                            # record schedule only if (i) valid packet downloaded (ii) packet fully downloaded
                            if (random_episodes-self.episode_step)<100: ##means the last 100 episodes
                                self.dqn_UL_schedule[i].append([self.episode_step, self.current_step, self.curr_UL_gen[i]])

                        if verbose:
                            print(f"user {i} age at the end is {self.UAV_age[i]}. old packet fully UL-new values curr_UL_gen[{i}] = {self.curr_UL_gen[i]}, comp_UL_gen[{i}] = {self.comp_UL_gen[i]}, RB_pending_UL[{i}] = {self.RB_pending_UL[i]}, assigned_RB_UL = {assigned_RB_UL}, remaining_RB_UL = {remaining_RB_UL}\n")
                    
                    else: # partial packet still going on
                        # self.UAV_age[i] = self.UAV_age[i] + 1 ## UAV age init 0 so will work. no if needed
                        
                        ## MOD
                        if (self.UAV_age[i] + 1)<=MAX_AGE:
                            self.UAV_age[i] = self.UAV_age[i] + 1
                        else:
                            self.UAV_age[i] = MAX_AGE 
                            self.UL_age_limit = self.UL_age_limit + 1
                            if verbose:
                                print(f"MAX_AGE for UL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}") 
                        
                        
                        if verbose:
                            print(f"user {i} age at the end is {self.UAV_age[i]}. old packet again partial UL-new values curr_UL_gen[{i}]={self.curr_UL_gen[i]}, comp_UL_gen[{i}] = {self.comp_UL_gen[i]}, RB_pending_UL[{i}] = {self.RB_pending_UL[i]}, assigned_RB_UL = {assigned_RB_UL}, remaining_RB_UL = {remaining_RB_UL}\n")
                        

            else: # remaining_RB_UL = 0
                # self.UAV_age[i] = self.UAV_age[i] + 1
                
                ## MOD
                if (self.UAV_age[i] + 1)<=MAX_AGE:
                    self.UAV_age[i] = self.UAV_age[i] + 1
                else:
                    self.UAV_age[i] = MAX_AGE 
                    self.UL_age_limit = self.UL_age_limit + 1
                    if verbose:
                        print(f"MAX_AGE for UL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}") 
                
                if verbose:
                    print(f"remaining_RB_UL = {remaining_RB_UL}, so device {i} not scheduled\n")
            
                    
        for i in self.user_list:
            if i not in upload_users:  
                if verbose:                
                    print(f"\nuser {i} age at the beginning is {self.UAV_age[i]}")
                # self.UAV_age[i] = self.UAV_age[i] + 1
                
                
                ## MOD
                if (self.UAV_age[i] + 1)<=MAX_AGE:
                    self.UAV_age[i] = self.UAV_age[i] + 1
                else:
                    self.UAV_age[i] = MAX_AGE
                    self.UL_age_limit = self.UL_age_limit + 1
                    if verbose:
                        print(f"MAX_AGE for UL {i}, DL_age_limit = {self.DL_age_limit}, UL_age_limit = {self.UL_age_limit}") 
                
                
                if verbose:
                    print(f"upload_users={upload_users}, so user {i} not selected. user {i} age at the end is {self.UAV_age[i]}\n")
                    
            ## uploading process ends
        
        # print(f"slot {self.current_step} ended with state {self._state}")        
                
      
        self._state = self.get_current_state() # update state after every action     
        for x in self._state:
            assert (x <= MAX_AGE) 
        dest_sum_age = np.sum(list(self.dest_age.values()))
        self.current_step += 1 # for next slot
        award = -dest_sum_age
        
        
        if verbose:
            print(f'current_step = {self.current_step-1} has award {award} for env {self.name}')
            print(f'next current_step will be = {self.current_step} with state {self.get_current_state()}') #, sample_time = {self.sample_time}')

        
        if self.current_step < self.MAX_STEPS + 1:        ## has to run for MAX_STEPS, i.e. an action has to be chosen MAX_STEPS times
            self._episode_ended = False
            return ts.transition(np.array([self._state], dtype=np.float32), reward = award, discount=1.0)
        else:
            # print(f'in terminate block') # will also reset the environment
            self._episode_ended = True
            # time_step.is_last() = True
            return ts.termination(np.array([self._state], dtype=np.float32), reward=award)