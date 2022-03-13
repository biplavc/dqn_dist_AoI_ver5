import time
import copy
import pickle
import os
import numpy as np 

path = "28dev_5UL_20DL\RP"

greedy_sch = pickle.load(open(path + "\\greedy_DL_schedule_28U.pickle", "rb"))
mad_sch = pickle.load(open(path + "\\mad_DL_schedule_28U.pickle", "rb"))
random_sch = pickle.load(open(path + "\\random_DL_schedule_28U.pickle", "rb"))
rr_sch = pickle.load(open(path + "\\rr_DL_schedule_28U.pickle", "rb"))

def generate_files(scheduler, links):
    """
    scheduler = type of scheduling
    links     = device wise communication instants
    """
    folder_name = "text_files/" + scheduler

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    for i in links:
        np_links = np.array(links[i])
        print(f"{i} \t{len(links[i])}", file = open(folder_name + "/dict.txt", "a"), flush = True)
        sub_folder = str(i)
        res = (np_links.flatten())
        res = [str(a) for a in res]
        res = (" " . join(res))
        print(f"{res}", file = open(folder_name + "/" + sub_folder + ".txt", "a"), flush = True)


def return_links(scheduler, current_ep):
    """
    sch = type of scheduler. e.g. greedy, MAD, random, round_robin
    current_ep = currrent_episode
    """

    if scheduler=="MAD":
        sch = mad_sch
    elif scheduler =="greedy":
        sch = greedy_sch
    elif scheduler == "random":
        sch = random_sch
    elif scheduler == "round_robin":
        sch = rr_sch

    MAX_EPS = 2001 ## number of times the code
    assert current_ep <=MAX_EPS+1

    SIM_DURATION = 1000 ## milliseconds the code runs in MATLAB

    links = {} ## tuple based keys
    ans_dup = {} ## char based keys

    for i in sch.keys():
        links[i] = []
        valid_links = sch[i]
        # print(f"valid_links = {valid_links}")
        for j in valid_links:
            # print(f"j={j}")
            # print(f"j[0]={j[0]}, current_ep={current_ep}")
            # time.sleep(3)
            if j[0]==current_ep:
                # print(f"ep is correct")
                # time.sleep(3)
                if j[1]<SIM_DURATION+1:
                # print(f"j={j}")
                    # print(f"SIM_DURATION is correct")
                    links[i].append([j[1], j[2]])
            else:
                # print(f"ep is wrong")
                pass

    ####### duplicate removal start

        # print(f"links[i] = {links[i]}")

        last_gen_time = 0
        to_remove = [] ## indexes of elements to be removed
        for ii in links[i]:
            # print(ii)
            curr_gen_time = ii[1]
            # print(f"curr_gen_time = {curr_gen_time}, last_gen_time = {last_gen_time}")
            if curr_gen_time == last_gen_time:
                # print(f"duplicate found for ii = {ii} , deleted item is {(x)[(x).index(ii)]}")
                to_remove.append(ii)
            last_gen_time = curr_gen_time

        for iii in to_remove:
            links[i].remove(iii)

    ####### duplicate removal end

    for k in links.keys():
        x,y = k[0], k[1]
        new_key = "t_" + str(x) + "_" + str(y)
        ans_dup[new_key] = links[k]

    generate_files(scheduler, ans_dup)
    # print(f"done")
    return ans_dup


ans = return_links("MAD", 13)
# ans = return_links("greedy", 1)
# ans = return_links("random", 1)
# ans = return_links("round_robin", 1)
# print(ans)