from tf_environment import *
from create_graph_1 import *
import collections
from collections import defaultdict
import itertools
import operator
import datetime
import sys
import math

# random.seed(42)

def find_omad_action(eval_env, slot): ## maximal age difference
    ## https://pynative.com/python-random-seed/
    seedValue = random.randrange(sys.maxsize)
    random.seed(seedValue) ## so that whenever randomness arises, different actions will be taken
    # print(seedValue) ## changes at each slot

    # print(f"eval_env.tx_rx_pairs = {eval_env.tx_rx_pairs}")
    ## the age is the difference of a user's age at the destination node and its age at the BS  
    
    ## select UL users start
   
    DL = {} ## key is user and the value is number of DL associated to that user
    
    for i in eval_env.tx_users:
        count = 0
        for j in eval_env.tx_rx_pairs:
            if i==j[0]:
                count = count + 1
        DL[i] = count     

    DL_copy = copy.deepcopy(DL)

    upload_users = []

    while len(upload_users) < len(eval_env.tx_users):
        # print(f"mad_user_pairs_excluded = {mad_user_pairs_excluded}, sampleDict_excluded_copy={sampleDict_excluded_copy}")
        
        ## get the ones with the max DL links

        itemMaxValue = max(DL_copy.items(), key=lambda x: x[1])
        listOfKeys_DL = list() ## users with max DL
        for key, value in DL_copy.items():
            if value == itemMaxValue[1]:
                listOfKeys_DL.append(key)

        ## for the ones with the max DL links, create a dict with value equal to UAV age

        listOfKeys_DL_dict = {}
        for i in listOfKeys_DL:
            listOfKeys_DL_dict[i] = eval_env.UAV_age[i]

        # print(f"listOfKeys_DL_dict = {listOfKeys_DL_dict}")

        ## for the ones with the max DL links, get the ones with the max UAV age

        listOfKeys_DL_dict_copy = copy.deepcopy(listOfKeys_DL_dict)

        itemMaxValue = max(listOfKeys_DL_dict_copy.items(), key=lambda x: x[1])
        listOfKeys = list() ## users with max age out of the ones with max DL
        for key, value in listOfKeys_DL_dict_copy.items():
            if value == itemMaxValue[1]:
                listOfKeys.append(key) ## max age for the max DL devices

        remaining_capacity = len(eval_env.tx_users) - len(upload_users)
        # print(f"remaining_capacity = {remaining_capacity}")
        if remaining_capacity > 0:
            upload_users.extend(random.sample(listOfKeys, len(listOfKeys)))

        # print(f"upload_users = {upload_users}")

        for items in listOfKeys:
            # print(f"no addition")
            del DL_copy[items]
            
    upload_users = upload_users[:eval_env.RB_total_UL]
    
    non_upload_users = []
    for i in eval_env.tx_users:
        if i not in upload_users:
            non_upload_users.append(i)
    
    ## select UL users end  
    
    ## select DL users start
    
    excluded_pairs = [] ## users who were not involved in the UL
    non_excluded_pairs = []
    
    for i in eval_env.tx_rx_pairs:
        if i[0] not in upload_users:
            excluded_pairs.append(i)
        else:
            non_excluded_pairs.append(i)
      
            
    if verbose:
        print(f"\n\nIMPTT - tx_users = {eval_env.tx_users}, tx_rx_pairs = {eval_env.tx_rx_pairs}, DL = {DL},  UAV_age = {eval_env.UAV_age}, dest_age = {eval_env.dest_age}, upload_users = {upload_users}, non_upload_users = {non_upload_users}, excluded_pairs = {excluded_pairs}, non_excluded_pairs = {non_excluded_pairs}") ## working
    
    ## mad_age for the excluded users
        
    mad_age_excluded_dest = {tuple(x):(eval_env.dest_age[tuple(x)] - eval_env.UAV_age[x[0]]) for x in excluded_pairs}   
    sampleDict_excluded_copy = copy.deepcopy(mad_age_excluded_dest)
    
    mad_age_non_excluded_dest = {tuple(x):(eval_env.dest_age[tuple(x)] - eval_env.UAV_age[x[0]]) for x in non_excluded_pairs}   
    sampleDict_non_excluded_copy = copy.deepcopy(mad_age_non_excluded_dest)
    
    mad_user_pairs_excluded = []  ## order excluded pairs based on age difference
    mad_user_pairs_non_excluded = [] ## order non-excluded pairs based on age difference
    
    if verbose:
        print(f"\n\nIMPTT - mad_age_excluded_dest = {mad_age_excluded_dest}, mad_age_non_excluded_dest = {mad_age_non_excluded_dest}\n")
        
    while len(mad_user_pairs_excluded) < len(mad_age_excluded_dest):
        # print(f"mad_user_pairs_excluded = {mad_user_pairs_excluded}, sampleDict_excluded_copy={sampleDict_excluded_copy}")
        itemMaxValue = max(sampleDict_excluded_copy.items(), key=lambda x: x[1])
        listOfKeys = list()
        for key, value in sampleDict_excluded_copy.items():
            if value == itemMaxValue[1]:
                listOfKeys.append(key)

        remaining_capacity = len(mad_age_excluded_dest) - len(mad_user_pairs_excluded)
        # print(f"remaining_capacity = {remaining_capacity}")
        if remaining_capacity > 0:
            mad_user_pairs_excluded.extend(random.sample(listOfKeys, len(listOfKeys)))
            # print(f"added")
        for items in listOfKeys:
            # print(f"no addition")
            del sampleDict_excluded_copy[items]
            
    while len(mad_user_pairs_non_excluded) < len(mad_age_non_excluded_dest):
        
        itemMaxValue = max(sampleDict_non_excluded_copy.items(), key=lambda x: x[1])
        listOfKeys = list()
        for key, value in sampleDict_non_excluded_copy.items():
            if value == itemMaxValue[1]:
                listOfKeys.append(key)

        remaining_capacity = len(mad_age_non_excluded_dest) - len(mad_user_pairs_non_excluded)
        if remaining_capacity > 0:
            mad_user_pairs_non_excluded.extend(random.sample(listOfKeys, len(listOfKeys)))
        for items in listOfKeys:
            del sampleDict_non_excluded_copy[items]
            
    mad_user_pairs = mad_user_pairs_excluded + mad_user_pairs_non_excluded
            
    mad_user_pairs_actual = mad_user_pairs[:eval_env.RB_total_DL]
    
    ## select DL users end
    
    download_user_pairs = mad_user_pairs_actual
    download_user_pairs_arr = []
    for i in download_user_pairs:
        download_user_pairs_arr.append(list(i))

    if verbose:
        print(f"IMPTT - mad_user_pairs_excluded = {mad_user_pairs_excluded}, mad_user_pairs_non_excluded = {mad_user_pairs_non_excluded}, mad_user_pairs = {mad_user_pairs}, download_user_pairs = {download_user_pairs}")

    ## select DL users end


    # if verbose:
    #     print(f"\nslot {slot} begins. age difference at the UAV is {mad_age_dest}, age at dest is {eval_env.dest_age}, age at UAV is {eval_env.UAV_age}.")  
        
    # time.sleep(10)
   
    ## pair selection for updating begins

 ############################################
       
    # actual_mad_action = None # this has to change
    
    # for action in range(eval_env.action_size):
    #     eval_action = eval_env.map_actions(action)
    #     # if verbose:
    #     #     print(f"\nlist(eval_action[0])={list(eval_action[0])},upload_users={upload_users},list(eval_action[1])={list(eval_action[1])},download_user_pairs={download_user_pairs},download_user_pairs_arr={download_user_pairs_arr}")
    #     #     time.sleep(10)
    #     if (list(eval_action[0]))==(upload_users) and (list(eval_action[1]))==(download_user_pairs_arr):
    #         actual_mad_action = action

    #         if verbose:
    #             print(f"upload_users={upload_users}, download_user_pairs={download_user_pairs_arr}, actual_mad_action = {actual_mad_action}")    
                
    #         break
        
    # assert actual_mad_action!=None
        
    return upload_users, download_user_pairs_arr     


def omad_links_UL_scheduling(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, MAX_STEPS):  ## maximal age difference
    
    # print(f"\nMAD started for {I} users , coverage = {drones_coverage} with update_loss_thresh = {packet_update_loss_thresh}, sample_loss_thresh = {packet_sample_loss_thresh}, periodicity = {periodicity}, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users}, RB_needed_UL = {RB_needed_UL}, RB_needed_DL = {RB_needed_DL} and {deployment} deployment", flush = True)
    
    print(f"\nomad_links_UL_scheduling started for {I} users , coverage = {drones_coverage} with packet_upload_loss_thresh = {packet_upload_loss_thresh}, packet_download_loss_thresh = {packet_download_loss_thresh}, periodicity = {periodicity}, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users},  RB_needed_UL = {RB_needed_UL}, RB_needed_DL = {RB_needed_DL}  and {deployment} deployment", file = open(folder_name + "/results.txt", "a"), flush = True)
    # do scheduling for MAX_STEPS random_episodes times and take the average
    final_step_rewards = []
    overall_ep_reward = []
    
    final_step_peak_rewards = []
    overall_ep_peak_reward = []
    
    final_step_UAV_rewards   = []
    
    # all_actions = [] # just saving all actions to see the distribution of the actions
    
    age_dist_UAV =  {} ## stores the episode ending (after MAX_STEPS) age for each user at UAV
    age_dist_dest  =  {} ## stores the episode ending (after MAX_STEPS) age for each pair at destination
    
    sample_time = {}
    for ii in periodicity.keys():
        sample_time[ii] = []
    
    attempt_upload = []
    success_upload = []
    attempt_download = []
    success_download = []
    
    total_packet_lost_upload = []
    total_packet_lost_download = []

    
    PDR_upload = []
    PDR_download = []

    unutilized_RB_UL = []
    unutilized_RB_DL = []
    
    mad_UL_schedule = {} ## key:([slot 1st DL completed, gen time of the packet], [slot 2nd DL completed, gen time of the packet], [...], [...])
    for i in tx_users:
        mad_UL_schedule[i] = [] ## list corresponding to each key

    mad_DL_schedule = {} ## key:([slot 1st UL completed, gen time of the packet], [slot 2nd UL completed, gen time of the packet], [...], [...])
    for i in tx_rx_pairs:
        mad_DL_schedule[tuple(i)] = [] ## list corresponding to each key

    best_episodes_average = {}
    best_episodes_peak = {}
     
    start_time = time.time()
    for ep in range(random_episodes): # how many times the random policy will be run, similar to episode
        if ep%log_interval_random ==0 and ep!=0:
            new_time = time.time()
            time_gap = round(new_time-start_time,2)
            rate = (new_time-start_time)/ep
            remaining_seconds = (random_episodes - ep)*rate
            x = datetime.datetime.now()
            finish_time = x + datetime.timedelta(seconds=remaining_seconds)
            print(f"oMAD links UL ep = {ep} took {time_gap} seconds from start, rate = {round((1/rate),2)} eps/sec, finish_time = {finish_time} ", flush = True)
        ep_reward = 0
        ep_peak_reward = 0

        eval_env = UAV_network(I, drones_coverage, "eval_net", folder_name, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, MAX_STEPS) ## will have just the eval env here

        eval_env.reset() # initializes age

        episode_wise_attempt_upload = 0
        episode_wise_success_upload = 0
        episode_wise_attempt_download = 0
        episode_wise_success_download = 0    
        
        packet_lost_upload = 0
        packet_lost_download = 0   

        
        if ep==0:
            print(f"\noMAD links UL scheduling and {deployment} placement with {I} users, coverage is {eval_env.act_coverage}, RB_total_UL is {RB_total_UL}, RB_total_DL = {RB_total_DL}, action space size is {eval_env.action_size} \n\n", file = open(folder_name + "/action_space.txt", "a"), flush = True)
            
            print(f"\noMAD links UL scheduling and {deployment} placement with {I} users, coverage is {eval_env.act_coverage}, RB_total_UL is {RB_total_UL}, RB_total_DL = {RB_total_DL}, action space size is {eval_env.action_size} ", file = open(folder_name + "/results.txt", "a"), flush = True)
            
            age_dist_UAV.update(eval_env.age_dist_UAV) # age_dist_UAV will have the appropriate keys and values
            age_dist_dest.update(eval_env.age_dist_dest) # age_dist_dest will have the appropriate keys and values ## working 
            
        eval_env.current_step  = 1            
        for i in eval_env.user_list:
            eval_env.tx_attempt_UAV[i].append(0) # 0 will be changed in _step for every attempt
            # age_dist_UAV[i].append(eval_env.UAV_age[i]) # eval_env.UAV_age has been made to 1 by now

        for i in eval_env.tx_rx_pairs:
            eval_env.tx_attempt_dest[tuple(i)].append(0) # 0 will be changed in _step for every attempt
            # age_dist_dest[tuple(i)].append(eval_env.dest_age[tuple(i)]) # eval_env.dest_age has been made to 1 by now
                
        # for i in range(eval_env.action_size):
        #     eval_env.preference[i].append(0) 
            
        for x in range(MAX_STEPS):
            # print(x)
            # print("inside MAD - ", x, " step started")      ## runs MAX_STEPS times       
            # selected_action = find_omad_action(eval_env, eval_env.current_step)
            # all_actions.append(selected_action)
            # # print("all_actions", type(all_actions))
            # eval_env.preference[selected_action][-1] = eval_env.preference[selected_action][-1] + 1
            # action = eval_env.map_actions(selected_action)
            
            upload_users, download_user_pairs = find_omad_action(eval_env, eval_env.current_step)


            if verbose:
                print(f"\n\ncurrent episode = {ep}\n")
                print(f"slot = {eval_env.current_step} for OMAD begins, upload_users = {upload_users}, download_user_pairs = {download_user_pairs}, BS_age = {eval_env.UAV_age}, dest_age = {eval_env.dest_age}\n")                
                
            ## downloading is done before uploading as downloading is done on the packets uploaded until the previous slot, so we say that packet uploaded in a slot cannot be downloaded in the same slot.
                
            ## download starts     
            
            remaining_RB_DL = eval_env.RB_total_DL
            
            for i in download_user_pairs:
                
                episode_wise_attempt_download = episode_wise_attempt_download + 1

                if verbose:
                    print(f"\ncurrent slot = {eval_env.current_step}, pair {i} age at the beginning is {eval_env.dest_age[tuple(i)]}")
                    
                received_SNR_download = getSNR(BS_location, user_locations[i[1]])
                
                if received_SNR_download < eval_env.packet_download_loss_thresh[tuple(i)]:
                    # print(f"PACKET LOSS - received_SNR_download = {received_SNR_download}, packet_download_loss_thresh[i] = {eval_env.packet_download_loss_thresh[tuple(i)]}")
                    packet_lost_download = packet_lost_download + 1
    
                if (remaining_RB_DL) > 0 and (received_SNR_download > eval_env.packet_download_loss_thresh[tuple(i)]): # scheduling only when RB available
                      
                    episode_wise_success_download = episode_wise_success_download + 1
                                       
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
                                
                                # record schedule only if (i) valid packet downloaded (ii) packet fully downloaded
                                if (random_episodes-ep)<100: ##means the last 100 episodes
                                    mad_DL_schedule[tuple(i)].append([ep, eval_env.current_step, eval_env.curr_DL_gen[tuple(i)]]) 
   
                            eval_env.comp_DL_gen[tuple(i)] = eval_env.curr_DL_gen[tuple(i)]
                            
                            
                            # new packet done
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
                            
                                # record schedule only if (i) valid packet sent (ii) packet fully downloaded
                                if (random_episodes-ep)<100: ##means the last 100 episodes
                                    mad_DL_schedule[tuple(i)].append([ep, eval_env.current_step, eval_env.curr_DL_gen[tuple(i)]]) 

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
                    
                episode_wise_attempt_upload = episode_wise_attempt_upload + 1
                
                received_SNR_upload = getSNR(BS_location, user_locations[i])
                
                if received_SNR_upload < eval_env.packet_upload_loss_thresh[i]:
                    # print(f"PACKET LOSS - received_SNR_upload = {received_SNR_upload}, packet_upload_loss_thresh[i] = {eval_env.packet_upload_loss_thresh[i]}")
                    packet_lost_upload = packet_lost_upload + 1
                    
                if (remaining_RB_UL > 0) and (received_SNR_upload > eval_env.packet_upload_loss_thresh[i]): ## only then scheduling can be done
                    
                    episode_wise_success_upload = episode_wise_success_upload + 1
                                            
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
                                
                                # record schedule only if (i) valid packet upload (ii) packet fully uploaded
                                if (random_episodes-ep)<100: ##means the last 100 episodes
                                    mad_UL_schedule[i].append([ep, eval_env.current_step, eval_env.curr_UL_gen[i]]) 
                            
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
                                
                                # record schedule only if (i) valid packet upload (ii) packet fully uploaded
                                if (random_episodes-ep)<100: ##means the last 100 episodes
                                    mad_UL_schedule[i].append([ep, eval_env.current_step, eval_env.curr_UL_gen[i]]) 

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
        
            eval_env.current_step += 1
            ep_reward = ep_reward + np.sum(list(eval_env.dest_age.values()))
            ep_peak_reward = ep_peak_reward + np.max(list(eval_env.dest_age.values()))
            # if verbose:
            # print(f"ep_peak_reward = {ep_peak_reward} for dest_age = {eval_env.dest_age}")
  
       
            if eval_env.current_step==MAX_STEPS:
                final_reward = np.sum(list(eval_env.dest_age.values()))
                # print("sum age at dest = ", final_reward)
                final_UAV_reward = np.sum(list(eval_env.UAV_age.values()))
                unutilized_RB_DL.append(remaining_RB_DL)
                unutilized_RB_UL.append(remaining_RB_UL)
                
                final_peak_reward = np.max(list(eval_env.dest_age.values()))
                
                if (random_episodes-ep)<100: ##means the last 100 episodes
                    best_episodes_average[ep] = ep_reward 
                    best_episodes_peak[ep] = ep_peak_reward 
   
    
        success_upload.append(episode_wise_success_upload) 
        success_download.append(episode_wise_success_download)
        attempt_upload.append(episode_wise_attempt_upload)
        attempt_download.append(episode_wise_attempt_download)     
        
        total_packet_lost_upload.append(packet_lost_upload)  
        total_packet_lost_download.append(packet_lost_download)
        
        final_step_rewards.append(final_reward)
        overall_ep_reward.append(ep_reward)
        final_step_UAV_rewards.append(final_UAV_reward)

        final_step_peak_rewards.append(final_peak_reward)
        overall_ep_peak_reward.append(ep_peak_reward)
        
        PDR_upload.append(episode_wise_success_upload/episode_wise_attempt_upload)
        PDR_download.append(episode_wise_success_download/episode_wise_attempt_download)            

        for i in eval_env.user_list:
            age_dist_UAV[i].append(eval_env.UAV_age[i]) 

        for i in eval_env.tx_rx_pairs:
            age_dist_dest[tuple(i)].append(eval_env.dest_age[tuple(i)]) 
            
       
        if verbose:
            print(f"age_dist_UAV = {age_dist_UAV}, age_dist_dest = {age_dist_dest}")
            print(f"results for step {eval_env.current_step} of episode {ep}")
            print(f"attempt_download = {attempt_download}")
            print(f"success_download = {success_download}")
            print(f"attempt_upload = {attempt_upload}")
            print(f"success_upload = {success_upload}")
            print(f"unutilized_RB_DL = {unutilized_RB_DL}")
            print(f"unutilized_RB_UL = {unutilized_RB_UL}")
            
            # time.sleep(10)
            print(f"\n*****************************************************\n")

    # print(f"mad_UL_schedule = {mad_UL_schedule}, mad_DL_schedule = {mad_DL_schedule}")
   
    pickle.dump(best_episodes_average, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_episodes_average.pickle", "wb"))
    pickle.dump(best_episodes_peak, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_best_episodes_peak.pickle", "wb"))
    
    pickle.dump(overall_ep_peak_reward, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_overall_ep_peak_reward.pickle", "wb"))    
    pickle.dump(final_step_peak_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_final_step_peak_rewards.pickle", "wb"))
    
    pickle.dump(overall_ep_reward, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_overall_ep_reward.pickle", "wb"))
    pickle.dump(final_step_UAV_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_final_step_UAV_rewards.pickle", "wb"))
    pickle.dump(final_step_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_final_step_rewards.pickle", "wb"))
    
    pickle.dump(age_dist_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_age_dist_UAV.pickle", "wb"))
    pickle.dump(age_dist_dest, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_age_dist_dest.pickle", "wb"))
    
    pickle.dump(eval_env.sample_time, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_sample_time.pickle", "wb"))
    
    pickle.dump(success_upload, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_success_upload.pickle", "wb"))
    pickle.dump(success_download, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_success_download.pickle", "wb"))
    pickle.dump(attempt_upload, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_attempt_upload.pickle", "wb"))
    pickle.dump(attempt_download, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_attempt_download.pickle", "wb"))
    
    pickle.dump(unutilized_RB_DL, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_unutilized_RB_DL.pickle", "wb"))
    pickle.dump(unutilized_RB_UL, open(folder_name + "/" + deployment + "/" + str(I) + "U_omad_unutilized_RB_UL.pickle", "wb"))
    
    pickle.dump(mad_UL_schedule, open(folder_name + "/" + deployment + "/" + "omad_UL_schedule_" + str(I) + "U.pickle", "wb"))    
    pickle.dump(mad_DL_schedule, open(folder_name + "/" + deployment + "/" + "omad_DL_schedule_" + str(I) + "U.pickle", "wb"))  
    
    print("\noMAD links UL scheduling ", deployment, " placement, ", I, " users, MAX_STEPS = ", MAX_STEPS, ", MEAN of overall_ep_reward = ", np.mean(overall_ep_reward), " MIN and MAX of overall_ep_reward = ", np.min(overall_ep_reward),", ", np.max(overall_ep_reward),  " ... MEAN of overall_ep_peak_reward = ", np.mean(overall_ep_peak_reward), " MIN and MAX of overall_ep_peak_reward = ", np.min(overall_ep_peak_reward),", ", np.max(overall_ep_peak_reward), ". Similarly for final_step_UAV_rewards - MEAN = ",{np.mean(final_step_UAV_rewards)}, ", MIN and MAX of final_step_UAV_rewards = ", np.min(final_step_UAV_rewards),", ", np.max(final_step_UAV_rewards)," end with final state of ", eval_env._state, " with shape ", np.shape(eval_env._state), ", min PDR_upload = ", {np.min(PDR_upload)}, ", max PDR_upload = ", {np.max(PDR_upload)}, ", min PDR_download = ", {np.min(PDR_download)}, ", max PDR_upload = ", {np.max(PDR_download)}, ", max_total_packet_lost_upload = ", np.max(total_packet_lost_upload), ", min_total_packet_lost_upload = ", np.min(total_packet_lost_upload), ", max_total_packet_lost_download = ", np.max(total_packet_lost_download), ", min_total_packet_lost_download = ", np.min(total_packet_lost_download))
    
    print("\noMAD links UL scheduling ", deployment, " placement, ", I, " users, MAX_STEPS = ", MAX_STEPS, ", MEAN of overall_ep_reward = ", np.mean(overall_ep_reward), " MIN and MAX of overall_ep_reward = ", np.min(overall_ep_reward),", ", np.max(overall_ep_reward),  " ... MEAN of overall_ep_peak_reward = ", np.mean(overall_ep_peak_reward), " MIN and MAX of overall_ep_peak_reward = ", np.min(overall_ep_peak_reward),", ", np.max(overall_ep_peak_reward), ". Similarly for final_step_UAV_rewards - MEAN = ",{np.mean(final_step_UAV_rewards)}, ", MIN and MAX of final_step_UAV_rewards = ", np.min(final_step_UAV_rewards),", ", np.max(final_step_UAV_rewards)," end with final state of ", eval_env._state, " with shape ", np.shape(eval_env._state), ", min PDR_upload = ", {np.min(PDR_upload)}, ", max PDR_upload = ", {np.max(PDR_upload)}, ", min PDR_download = ", {np.min(PDR_download)}, ", max PDR_upload = ", {np.max(PDR_download)}, ", max_total_packet_lost_upload = ", np.max(total_packet_lost_upload), ", min_total_packet_lost_upload = ", np.min(total_packet_lost_upload), ", max_total_packet_lost_download = ", np.max(total_packet_lost_download), ", min_total_packet_lost_download = ", np.min(total_packet_lost_download), file = open(folder_name + "/results.txt", "a"), flush = True)
    
    sorted_best_episodes_average = sorted(best_episodes_average.items(), key=lambda x: x[1])
    sorted_best_episodes_peak = sorted(best_episodes_peak.items(), key=lambda x: x[1])
    
    print(f"oMAD links UL best_episodes_average = {sorted_best_episodes_average}", file = open(folder_name + "/best_episodes.txt", "a"), flush = True)
    print(f"\n\n\n", file = open(folder_name + "/best_episodes.txt", "a"), flush = True)
    print(f"oMAD links UL sorted_best_episodes_peak = {sorted_best_episodes_peak}", file = open(folder_name + "/best_episodes.txt", "a"), flush = True)
    print(f"\n\n\n ------------------------------------------------", file = open(folder_name + "/best_episodes.txt", "a"), flush = True)

   
    assert(len(final_step_rewards)==len(final_step_rewards))
    return overall_ep_reward, final_step_rewards, np.mean(overall_ep_reward)