from tf_environment import *
from create_graph_1 import *
import collections
from collections import defaultdict
import itertools
import operator
from datetime import datetime
import sys
import math

# random.seed(42)

def find_lrb_action(eval_env, slot): ## maximal age difference
    ## https://pynative.com/python-random-seed/
    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue) ## so that whenever randomness arises, different actions will be taken
    # print(seedValue) ## changes at each slot

    # print(f"eval_env.tx_rx_pairs = {eval_env.tx_rx_pairs}")
    ## the age is the difference of a user's age at the destination node and its age at the BS    
    ## random select DL pairs start

    # lrb_age_dest = {tuple(x):(eval_env.dest_age[tuple(x)] - eval_env.UAV_age[x[0]]) for x in eval_env.tx_rx_pairs}
    sampleDict_copy = copy.deepcopy(eval_env.RB_needed_DL)
    
    lrb_user_pairs = []
    while len(lrb_user_pairs) < len(eval_env.tx_rx_pairs):

        itemMaxValue = min(sampleDict_copy.items(), key=lambda x: x[1])
        listOfKeys = list()
        for key, value in sampleDict_copy.items():
            if value == itemMaxValue[1]:
                listOfKeys.append(key)

        remaining_capacity = len(eval_env.tx_rx_pairs) - len(lrb_user_pairs)
        if remaining_capacity > 0:
            lrb_user_pairs.extend(random.sample(listOfKeys, len(listOfKeys)))
        for items in listOfKeys:
            del sampleDict_copy[items]

    lrb_user_pairs_actual = lrb_user_pairs[:eval_env.RB_total_DL]
 
    ## random select DL pairs end
    
    download_user_pairs = lrb_user_pairs_actual
    download_user_pairs_arr = []
    for i in download_user_pairs:
        download_user_pairs_arr.append(list(i))

    ## random select UL users start
    upload_users = []
    sampleDict_copy = copy.deepcopy(eval_env.RB_needed_UL)

    while len(upload_users) < len(eval_env.tx_users):

        itemMaxValue = min(sampleDict_copy.items(), key=lambda x: x[1])
        listOfKeys = list()
        for key, value in sampleDict_copy.items():
            if value == itemMaxValue[1]:
                listOfKeys.append(key)

        remaining_capacity = len(eval_env.tx_users) - len(upload_users)
        if remaining_capacity > 0:
            upload_users.extend(random.sample(listOfKeys, len(listOfKeys)))
        for items in listOfKeys:
            del sampleDict_copy[items]
            
    upload_users = upload_users[:eval_env.RB_total_UL]
            
    ## random select UL users end

    if verbose:
        print(f"\nslot {slot} begins. age at dest is {eval_env.dest_age}, age at UAV is {eval_env.UAV_age}.")  
        
    # time.sleep(10)
   
    ## pair selection for updating begins

 ############################################
       
    actual_lrb_action = None # this has to change
    
    for action in range(eval_env.action_size):
        eval_action = eval_env.map_actions(action)
        # if verbose:
        #     print(f"\nlist(eval_action[0])={list(eval_action[0])},upload_users={upload_users},list(eval_action[1])={list(eval_action[1])},download_user_pairs={download_user_pairs},download_user_pairs_arr={download_user_pairs_arr}")
        #     time.sleep(10)
        if (list(eval_action[0]))==(upload_users) and (list(eval_action[1]))==(download_user_pairs_arr):
            actual_lrb_action = action

            if verbose:
                print(f"upload_users={upload_users}, download_user_pairs={download_user_pairs_arr}, actual_lrb_action = {actual_lrb_action}")    
                
            break
        
    assert actual_lrb_action!=None
        
    return actual_lrb_action    


def lrb_scheduling(I, drones_coverage, folder_name, deployment, packet_update_loss, packet_sample_loss, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL):  ## maximal age difference
    
    print(f"\nLRB started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity}, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users}, RB_needed_UL = {RB_needed_UL}, RB_needed_DL = {RB_needed_DL} and {deployment} deployment")
    
    print(f"\nLRB started for {I} users , coverage = {drones_coverage} with update_loss = {packet_update_loss}, sample_loss = {packet_sample_loss}, periodicity = {periodicity}, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users},  RB_needed_UL = {RB_needed_UL}, RB_needed_DL = {RB_needed_DL}  and {deployment} deployment", file = open(folder_name + "/results.txt", "a"), flush = True)
    # do scheduling for MAX_STEPS random_episodes times and take the average
    final_step_rewards = []
    overall_ep_reward = []
    final_step_UAV_rewards   = []
    
    all_actions = [] # just saving all actions to see the distribution of the actions
    
    age_dist_UAV =  {} ## stores the episode ending (after MAX_STEPS) age for each user at UAV
    age_dist_dest  =  {} ## stores the episode ending (after MAX_STEPS) age for each pair at destination
    
    sample_time = {}
    for ii in periodicity.keys():
        sample_time[ii] = []
    
    attempt_sample = []
    success_sample = []
    attempt_update = []
    success_update = []

    unutilized_RB_UL = []
    unutilized_RB_DL = []
    
    
    for ep in range(random_episodes): # how many times this policy will be run, similar to episode

        ep_reward = 0
        eval_env = UAV_network(I, drones_coverage, "eval_net", folder_name, packet_update_loss, packet_sample_loss, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL) ## will have just the eval env here

        eval_env.reset() # initializes age

        episode_wise_attempt_sample = 0
        episode_wise_success_sample = 0
        episode_wise_attempt_update = 0
        episode_wise_success_update = 0    

        
        if ep==0:
            print(f"\nLRB scheduling and {deployment} placement with {I} users, coverage is {eval_env.act_coverage}, RB_total_UL is {RB_total_UL}, RB_total_DL = {RB_total_DL}, action space size is {eval_env.action_size} \n\n", file = open(folder_name + "/action_space.txt", "a"), flush = True)
            
            print(f"\nLRB scheduling and {deployment} placement with {I} users, coverage is {eval_env.act_coverage}, RB_total_UL is {RB_total_UL}, RB_total_DL = {RB_total_DL}, action space size is {eval_env.action_size} ", file = open(folder_name + "/results.txt", "a"), flush = True)
            
            age_dist_UAV.update(eval_env.age_dist_UAV) # age_dist_UAV will have the appropriate keys and values
            age_dist_dest.update(eval_env.age_dist_dest) # age_dist_dest will have the appropriate keys and values ## working 
            
        eval_env.current_step  = 1            
        for i in eval_env.user_list:
            eval_env.tx_attempt_UAV[i].append(0) # 0 will be changed in _step for every attempt
            # age_dist_UAV[i].append(eval_env.UAV_age[i]) # eval_env.UAV_age has been made to 1 by now

        for i in eval_env.tx_rx_pairs:
            eval_env.tx_attempt_dest[tuple(i)].append(0) # 0 will be changed in _step for every attempt
            # age_dist_dest[tuple(i)].append(eval_env.dest_age[tuple(i)]) # eval_env.dest_age has been made to 1 by now
                
        for i in range(eval_env.action_size):
            eval_env.preference[i].append(0) 
            
        for x in range(MAX_STEPS):
            # print(x)
            # print("inside lrb - ", x, " step started")      ## runs MAX_STEPS times       
            selected_action = find_lrb_action(eval_env, eval_env.current_step)
            all_actions.append(selected_action)
            # print("all_actions", type(all_actions))
            eval_env.preference[selected_action][-1] = eval_env.preference[selected_action][-1] + 1
            action = eval_env.map_actions(selected_action)
            
            download_user_pairs = list(action[1])
            upload_users = list(action[0])

                
            if verbose:
                # print(f"\n\ncurrent episode = {ep}\n")
                print(f"lrb selection is {selected_action}, actual action is {action}, upload_users = {upload_users}, download_user_pairs = {download_user_pairs}, BS_age = {eval_env.UAV_age}, dest_age = {eval_env.dest_age}\n") 
                
            ## downloading is done before uploading as downloading is done on the packets uploaded until the previous slot, so we say that packet uploaded in a slot cannot be downloaded in the same slot.
                
            ## download starts     
            
            remaining_RB_DL = eval_env.RB_total_DL
            
            for i in download_user_pairs:
                if verbose:
                    print(f"\ncurrent slot = {eval_env.current_step}, pair {i} age at the beginning is {eval_env.dest_age[tuple(i)]}")

                if remaining_RB_DL > 0: # scheduling only when RB available
                    
                    if eval_env.comp_UL_gen[i[0]] == -1: # means no packet has been uploaded yet by the source
                        if verbose:
                            print(f'pair {i}, i.e. user {i[0]} has no data yet at the BS so empty packet will be sent')
                        
                    # if (eval_env.comp_DL_gen[tuple(i)]!=eval_env.comp_UL_gen[i[0]]): # and (eval_env.comp_DL_gen[tuple(i)]!=eval_env.curr_DL_gen[tuple(i)]):  ## ?? which one needed ?
                        
                    # before DL a new packet, check if it has already been DL. Might happen when no new uploaded but DL keeps getting scheduled
                    # true means the packet currently to be DL (comp_UL_gen[i[0]]) is different than the prev DL (comp_DL_gen[tuple(i)])
                    

                    assert (eval_env.comp_DL_gen[tuple(i)] <= eval_env.comp_UL_gen[i[0]]) # as BS will have newer or same packet that has been recently DL to dest completely.
                            
                    if eval_env.RB_pending_DL[tuple(i)]==0: # prev packet fully downloaded in prev attempt, so download the recently uploaded packet    
                        if verbose:
                            print(f"pair {i} completed DL in prev attempt so new packet DL. old values-curr_DL_gen[{i}]={eval_env.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={eval_env.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={eval_env.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={eval_env.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={eval_env.RB_pending_DL[tuple(i)]}, remaining_RB_DL={remaining_RB_DL}")       
                                                                
                        
                        eval_env.curr_DL_gen[tuple(i)]   = eval_env.comp_UL_gen[i[0]] # already uploaded packet
                        eval_env.RB_pending_DL[tuple(i)] = eval_env.RB_needed_DL[tuple(i)] # as new packet
                        assigned_RB_DL  = min(remaining_RB_DL, eval_env.RB_pending_DL[tuple(i)])
                        remaining_RB_DL = remaining_RB_DL - assigned_RB_DL
                        eval_env.RB_pending_DL[tuple(i)] = eval_env.RB_pending_DL[tuple(i)] - assigned_RB_DL # new pending after assignment
                        # new packet details ended

                        assert(assigned_RB_DL>-1 and eval_env.RB_pending_DL[tuple(i)]>-1)
                        
                        if (eval_env.comp_DL_gen[tuple(i)]==eval_env.curr_DL_gen[tuple(i)]) and (eval_env.comp_DL_gen[tuple(i)]!=-1): # comp_DL_gen is from the prev completed attempt and curr_DL_gen is newly set, so comparision correct. will only happen when DL a new packet
                            if verbose:
                                print(f"packet being DL has been already DL")  
                                                
                        if eval_env.RB_pending_DL[tuple(i)] == 0: # means packet was fully downloaded in this slot
                            # finish off DL of new packet
                            
                            if eval_env.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                                eval_env.dest_age[tuple(i)] = eval_env.current_step


                            else: ## current packet started DL after BS had a packet
                                eval_env.dest_age[tuple(i)] = eval_env.current_step - eval_env.curr_DL_gen[tuple(i)] # age change after packet fully sent
   
                            eval_env.comp_DL_gen[tuple(i)] = eval_env.curr_DL_gen[tuple(i)]
                            # older packet done
                            if verbose:
                                print(f"pair {i} age at the end is {eval_env.dest_age[tuple(i)]}. completed DL new pack in same slot. new values-curr_DL_gen[{i}]={eval_env.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={eval_env.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={eval_env.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={eval_env.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={eval_env.RB_pending_DL[tuple(i)]},assigned_RB_DL = {assigned_RB_DL}, remaining_RB_DL={remaining_RB_DL}\n")
                            
                        else: # eval_env.RB_pending_DL[tuple(i)] != 0. new packet incomplete DL
                                
                            if eval_env.comp_DL_gen[tuple(i)] == -1: ## no packet DL till now 
                                eval_env.dest_age[tuple(i)] = eval_env.current_step  
                                
                                                          
                            else: ## at least a packet had been downloaded
                                eval_env.dest_age[tuple(i)] = eval_env.dest_age[tuple(i)] + 1 # age change after packet partial sent
                                
                            if verbose:
                                print(f"pair {i} age at the end is {eval_env.dest_age[tuple(i)]}. partial DL new pack in current attempt. new values-curr_DL_gen[{i}]={eval_env.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={eval_env.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={eval_env.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={eval_env.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={eval_env.RB_pending_DL[tuple(i)]},assigned_RB_DL = {assigned_RB_DL}, remaining_RB_DL={remaining_RB_DL}\n") 
                        
                    else: ## eval_env.RB_pending_DL[tuple(i)]!=0
                    # old packet from prev attempts is still being downloaded. part of older packet and part of newer packet will not be done together even if RBs available. Attempt to transmit the older packet fully here
                        
                        if verbose:
                            print(f"pair {i} continue incomplete DL from prev. old values-curr_DL_gen[{i}]={eval_env.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={eval_env.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={eval_env.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={eval_env.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={eval_env.RB_pending_DL[tuple(i)]}, remaining_RB_DL={remaining_RB_DL}")
                        
                        assigned_RB_DL  = min(remaining_RB_DL, eval_env.RB_pending_DL[tuple(i)])
                        remaining_RB_DL = remaining_RB_DL - assigned_RB_DL
                        eval_env.RB_pending_DL[tuple(i)] = eval_env.RB_pending_DL[tuple(i)] - assigned_RB_DL
                        
                        
                        if eval_env.RB_pending_DL[tuple(i)] == 0: # means old packet was fully downloaded in this slot
                            # finish off DL of selected packet
                            if eval_env.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                                eval_env.dest_age[tuple(i)] = eval_env.current_step
                            else: ## current packet started DL after BS had a packet
                                eval_env.dest_age[tuple(i)] = eval_env.current_step - eval_env.curr_DL_gen[tuple(i)] # age change after packet fully sent
                                eval_env.comp_DL_gen[tuple(i)] = eval_env.curr_DL_gen[tuple(i)]
                            # older packet done

                            if verbose:
                                print(f"pair {i} age at the end is {eval_env.dest_age[tuple(i)]}. old packet fully DL. new values-curr_DL_gen[{i}]={eval_env.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={eval_env.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={eval_env.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={eval_env.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={eval_env.RB_pending_DL[tuple(i)]},assigned_RB_DL = {assigned_RB_DL}, remaining_RB_DL={remaining_RB_DL}\n")
                                
                        else: # eval_env.RB_pending_DL[tuple(i)] != 0: # means old packet wasn't fully downloaded in this slot
                            if eval_env.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                                eval_env.dest_age[tuple(i)] = eval_env.current_step
                            else: ## current packet started DL after BS had a packet
                                eval_env.dest_age[tuple(i)] = eval_env.dest_age[tuple(i)] + 1 # age change after packet fully sent
                                if verbose:
                                    print(f"pair {i} age at the end is {eval_env.dest_age[tuple(i)]}. old packet again partial DL. new values-curr_DL_gen[{i}]={eval_env.curr_DL_gen[tuple(i)]}, comp_DL_gen[{i}]={eval_env.comp_DL_gen[tuple(i)]}, curr_UL_gen[{i[0]}]={eval_env.curr_UL_gen[i[0]]}, comp_UL_gen[{i[0]}]={eval_env.comp_UL_gen[i[0]]}, RB_pending_DL[{i}]={eval_env.RB_pending_DL[tuple(i)]},assigned_RB_DL = {assigned_RB_DL}, remaining_RB_DL={remaining_RB_DL}\n")  
                        
                        assert(assigned_RB_DL>-1 and remaining_RB_DL>-1 and eval_env.RB_pending_DL[tuple(i)]>-1)
                           

                else: # no RB remaining
                    if eval_env.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                        eval_env.dest_age[tuple(i)] = eval_env.current_step
                    else: ## current packet started DL after BS had a packet
                        eval_env.dest_age[tuple(i)] = eval_env.dest_age[tuple(i)] + 1 # age change after packet fully sent                    
                    if verbose:
                        print(f"remaining_RB_DL = {remaining_RB_DL}, so pair {i} not scheduled. pair {i} age at the end is {eval_env.dest_age[tuple(i)]}\n")
 
            for i in eval_env.tx_rx_pairs: 
                if i not in download_user_pairs:
                    if verbose:                
                        print(f"\npair {i} age at the beginning is {eval_env.dest_age[tuple(i)]}")
                    if eval_env.curr_DL_gen[tuple(i)] == -1: ## current packet started DL when BS had nothing 
                        eval_env.dest_age[tuple(i)] = eval_env.current_step
                    else: ## current packet started DL after BS had a packet
                        eval_env.dest_age[tuple(i)] = eval_env.dest_age[tuple(i)] + 1 # age change after packet fully sent                      
                    
                    if verbose:
                        print(f"download_user_pairs={download_user_pairs}, so pair {i} not selected. pair {i} age at the end is {eval_env.dest_age[tuple(i)]}\n")
     
            ## download ends     
                
            ## uploading process start
                
            remaining_RB_UL = eval_env.RB_total_UL ## all RBs available in the beginning of the slot
                        
            for i in upload_users:
                if verbose:
                    print(f"\nuser {i} age at the beginning is {eval_env.UAV_age[i]} with period {eval_env.periodicity[i]}")
                
                if remaining_RB_UL > 0: ## only then scheduling can be done
                                            
                    if eval_env.RB_pending_UL[i]==0: ## this user has completed the prev packet in its prev attempt, switch to a new packet now. both sample and try to send the new packet. also for the first slot, this wil be true  
                        
                        if verbose:
                            print(f"user {i} completed UL in its prev attempt. old values-curr_UL_gen[{i}] = {eval_env.curr_UL_gen[i]}, comp_UL_gen[{i}] = {eval_env.comp_UL_gen[i]}, RB_pending_UL[{i}] = {eval_env.RB_pending_UL[i]}, remaining_RB_UL = {remaining_RB_UL}.")
                        
                        # new packet details started
                        if eval_env.current_step%eval_env.periodicity[i]==0: # generate at will has period 1 so %=0
                            last_pack_generated = eval_env.current_step  # generation time of the packet sampled now
                            # print(f"last_pack_generated 1 = {last_pack_generated}. eval_env.current_step%eval_env.periodicity[i]={eval_env.current_step%eval_env.periodicity[i]}")
                        else:
                            last_pack_generated = eval_env.periodicity[i]*max(math.floor(eval_env.current_step/eval_env.periodicity[i]), 1) ## max added so that the result is never 0 as our slots start from 1. so periodicity of 2 means 2,4,6,8
                            # print(f"last_pack_generated 2 = {last_pack_generated}. eval_env.current_step%eval_env.periodicity[i]={eval_env.current_step%eval_env.periodicity[i]}")

                        
                        if eval_env.current_step >= last_pack_generated: # else will stay -1
                            eval_env.curr_UL_gen[i]  = last_pack_generated # eval_env.current_step # sample at will, so new packet generated and sampled only if current time and last generated time is feasible
                        
                        if verbose:
                            print(f"last_pack_generated = {last_pack_generated}")                    
                                                   
                        eval_env.RB_pending_UL[i] = eval_env.RB_needed_UL[i] # as new packet
                        assigned_RB_UL  = min(remaining_RB_UL, eval_env.RB_pending_UL[i])
                        remaining_RB_UL = remaining_RB_UL - assigned_RB_UL
                        eval_env.RB_pending_UL[i] = eval_env.RB_pending_UL[i] - assigned_RB_UL # new pending after assignment
                        assert(assigned_RB_UL>-1 and remaining_RB_UL>-1 and eval_env.RB_pending_UL[i]>-1)
                        # new packet details ended
                        
                        if eval_env.RB_pending_UL[i] == 0: # means packet was fully uploaded in this slot
                            # finish off details of completed packet
                            if eval_env.curr_UL_gen[i] == -1: ## current packet started UL when device had nothing 
                                eval_env.UAV_age[i] = eval_env.current_step
                            else: ## current packet started UL after device had a packet
                                eval_env.UAV_age[i] = eval_env.current_step - eval_env.curr_UL_gen[i] # age change after packet fully sent                        
                                eval_env.comp_UL_gen[i] = eval_env.curr_UL_gen[i] # no need of multiple cases like DL
                            
                            # new packet done
                            
                            if verbose:
                                print(f"user {i} age at the end is {eval_env.UAV_age[i]}. new packet fully UL in same slot-new values curr_UL_gen[{i}] = {eval_env.curr_UL_gen[i]}, comp_UL_gen[{i}] = {eval_env.comp_UL_gen[i]}, RB_pending_UL[{i}] = {eval_env.RB_pending_UL[i]}, assigned_RB_UL = {assigned_RB_UL}, remaining_RB_UL = {remaining_RB_UL}\n")
                        
                        else: # means packet was not fully uploaded in this slot
                            eval_env.UAV_age[i] = eval_env.UAV_age[i] + 1
                            if verbose:
                                print(f"user {i} age at the end is {eval_env.UAV_age[i]}. new packet partial UL-new values curr_UL_gen[{i}] = {eval_env.curr_UL_gen[i]}, comp_UL_gen[{i}] = {eval_env.comp_UL_gen[i]}, RB_pending_UL[{i}] = {eval_env.RB_pending_UL[i]}, assigned_RB_UL = {assigned_RB_UL}, remaining_RB_UL = {remaining_RB_UL}\n")
                    
                    else: # old packet from prev attempts is still being sent. part of older packet and part of newer packet will not be done together even if RBs available. Attempt to transmit the older packet fully
                        
                        if verbose:
                            print(f"continue incomplete UL from prev-old values-curr_UL_gen[{i}] = {eval_env.curr_UL_gen[i]}, comp_UL_gen[{i}] = {eval_env.comp_UL_gen[i]}, RB_pending_UL[{i}] = {eval_env.RB_pending_UL[i]}, remaining_RB_UL = {remaining_RB_UL}")
                        
                        assigned_RB_UL = min(remaining_RB_UL, eval_env.RB_pending_UL[i])
                        remaining_RB_UL = remaining_RB_UL - assigned_RB_UL
                        eval_env.RB_pending_UL[i] = eval_env.RB_pending_UL[i] - assigned_RB_UL
                        assert(assigned_RB_UL>-1 and remaining_RB_UL>-1 and eval_env.RB_pending_UL[i]>-1)

                        if eval_env.RB_pending_UL[i] == 0: # means the partial packet was fully uploaded in this slot
                            # finish off details of completed packet
                            if eval_env.curr_UL_gen[i] == -1: ## current packet started UL when device had nothing 
                                eval_env.UAV_age[i] = eval_env.current_step
                            else: ## current packet started UL after device had a packet
                                eval_env.UAV_age[i] = eval_env.current_step - eval_env.curr_UL_gen[i] # age change after packet fully sent                        
                                eval_env.comp_UL_gen[i] = eval_env.curr_UL_gen[i] # no need of multiple cases like DL

                            if verbose:
                                print(f"user {i} age at the end is {eval_env.UAV_age[i]}. old packet fully UL-new values curr_UL_gen[{i}] = {eval_env.curr_UL_gen[i]}, comp_UL_gen[{i}] = {eval_env.comp_UL_gen[i]}, RB_pending_UL[{i}] = {eval_env.RB_pending_UL[i]}, assigned_RB_UL = {assigned_RB_UL}, remaining_RB_UL = {remaining_RB_UL}\n")
                        
                        else: # partial packet still going on
                            eval_env.UAV_age[i] = eval_env.UAV_age[i] + 1 ## UAV age init 0 so will work. no if needed
                            if verbose:
                                print(f"user {i} age at the end is {eval_env.UAV_age[i]}. old packet again partial UL-new values curr_UL_gen[{i}]={eval_env.curr_UL_gen[i]}, comp_UL_gen[{i}] = {eval_env.comp_UL_gen[i]}, RB_pending_UL[{i}] = {eval_env.RB_pending_UL[i]}, assigned_RB_UL = {assigned_RB_UL}, remaining_RB_UL = {remaining_RB_UL}\n")
                            

                else: # remaining_RB_UL = 0
                    eval_env.UAV_age[i] = eval_env.UAV_age[i] + 1
                    if verbose:
                        print(f"remaining_RB_UL = {remaining_RB_UL}, so device {i} not scheduled\n")
              
                        
            for i in eval_env.user_list:
                if i not in upload_users:  
                    if verbose:                
                        print(f"\nuser {i} age at the beginning is {eval_env.UAV_age[i]}")
                    eval_env.UAV_age[i] = eval_env.UAV_age[i] + 1
                    if verbose:
                        print(f"upload_users={upload_users}, so user {i} not selected. user {i} age at the end is {eval_env.UAV_age[i]}\n")
                        
                ## uploading process ends
                
            if verbose:
                print(f"\nslot {eval_env.current_step} over. curr_UL_gen = {eval_env.curr_UL_gen}, curr_DL_gen = {eval_env.curr_DL_gen}, comp_UL_gen = {eval_env.comp_UL_gen}, comp_DL_gen = {eval_env.comp_DL_gen}, RB_pending_UL = {eval_env.RB_pending_UL}, RB_pending_DL = {eval_env.RB_pending_DL}, BS_age = {eval_env.UAV_age}, dest_age = {eval_env.dest_age}\n")   
                
                if eval_env.curr_UL_gen==eval_env.comp_UL_gen and eval_env.curr_DL_gen==eval_env.comp_DL_gen:
                    print(f"no informative packet pending")

            if eval_env.current_step==MAX_STEPS:
                final_reward = np.sum(list(eval_env.dest_age.values()))
                # print("sum age at dest = ", final_reward)
                final_UAV_reward = np.sum(list(eval_env.UAV_age.values()))
                unutilized_RB_DL.append(remaining_RB_DL)
                unutilized_RB_UL.append(remaining_RB_UL)
                
            eval_env.current_step += 1
            ep_reward = ep_reward + np.sum(list(eval_env.dest_age.values()))

           
        attempt_sample.append(episode_wise_attempt_sample)
        success_sample.append(episode_wise_success_sample)
        attempt_update.append(episode_wise_attempt_update)
        success_update.append(episode_wise_success_update)
        
        final_step_rewards.append(final_reward)
        overall_ep_reward.append(ep_reward)
        final_step_UAV_rewards.append(final_UAV_reward)
        

        for i in eval_env.user_list:
            age_dist_UAV[i].append(eval_env.UAV_age[i]) 

        for i in eval_env.tx_rx_pairs:
            age_dist_dest[tuple(i)].append(eval_env.dest_age[tuple(i)]) 
            
       
        if verbose:
            print(f"age_dist_UAV = {age_dist_UAV}, age_dist_dest = {age_dist_dest}")
            print(f"results for step {eval_env.current_step} of episode {ep}")
            print(f"attempt_sample = {attempt_sample}")
            print(f"success_sample = {success_sample}")
            print(f"attempt_update = {attempt_update}")
            print(f"success_update = {success_update}")
            print(f"unutilized_RB_DL = {unutilized_RB_DL}")
            print(f"unutilized_RB_UL = {unutilized_RB_UL}")
            
            # time.sleep(10)
            print(f"\n*****************************************************\n")

    pickle.dump(overall_ep_reward, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_overall_ep_reward.pickle", "wb"))
    pickle.dump(final_step_UAV_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_final_step_UAV_rewards.pickle", "wb"))
    pickle.dump(final_step_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_final_step_rewards.pickle", "wb"))

    pickle.dump(age_dist_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_age_dist_UAV.pickle", "wb"))
    pickle.dump(age_dist_dest, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_age_dist_dest.pickle", "wb"))
    
    pickle.dump(eval_env.sample_time, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_sample_time.pickle", "wb"))
    
    pickle.dump(attempt_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_attempt_sample.pickle", "wb"))
    pickle.dump(success_sample, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_success_sample.pickle", "wb"))
    pickle.dump(attempt_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_attempt_update.pickle", "wb"))
    pickle.dump(success_update, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_success_update.pickle", "wb"))
    
    pickle.dump(unutilized_RB_DL, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_unutilized_RB_DL.pickle", "wb"))
    pickle.dump(unutilized_RB_UL, open(folder_name + "/" + deployment + "/" + str(I) + "U_lrb_unutilized_RB_UL.pickle", "wb"))
    
    
    print("\nlrb scheduling ", deployment, " placement, ", I, " users - MEAN of final_step_rewards = ", np.mean(final_step_rewards), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(overall_ep_reward), ". Similarly for final_step_UAV_rewards - MEAN =",np.mean(final_step_UAV_rewards), ", MIN and MAX of final_step_UAV_rewards = ", np.min(final_step_UAV_rewards),", ", np.max(final_step_UAV_rewards)," end with final state of ", eval_env._state, " with shape ", np.shape(eval_env._state))
    
    print("\nlrb scheduling ", deployment, " placement, ", I, " users - MEAN of final_step_rewards = ", np.mean(final_step_rewards), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", np.mean(overall_ep_reward), ". Similarly for final_step_UAV_rewards - MEAN =",np.mean(final_step_UAV_rewards), ", MIN and MAX of final_step_UAV_rewards = ", np.min(final_step_UAV_rewards),", ", np.max(final_step_UAV_rewards)," end with final state of ", eval_env._state, " with shape ", np.shape(eval_env._state), file = open(folder_name + "/results.txt", "a"), flush = True)
   
    assert(len(final_step_rewards)==len(final_step_rewards))
    return overall_ep_reward, final_step_rewards, all_actions
    