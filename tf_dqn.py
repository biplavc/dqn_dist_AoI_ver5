from tf_environment import *
from create_graph_1 import *
import datetime 

# if comet:
#     from main_tf import experiment
random.seed(42)
np.random.seed(42)
# tf.random.set_seed(42)

# source - https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())

def tf_dqn(I, drones_coverage, folder_name, deployment, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T):   
    
    mode = "train" ## "train" "deploy"
    
    policy_dir = os.path.join(folder_name, 'policy')

    if mode == "train":
   
        all_actions = []    ## save all actions over all steps of all episodes  
        print(f"\nDQN started for started for {I} users , coverage = {drones_coverage} with packet_upload_loss_thresh = {packet_upload_loss_thresh}, packet_download_loss_thresh = {packet_download_loss_thresh}, periodicity = {periodicity}, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users},  RB_needed_UL = {RB_needed_UL}, RB_needed_DL = {RB_needed_DL}  and {deployment} deployment")
        
        print(f"\nDQN started for started for {I} users , coverage = {drones_coverage} with packet_upload_loss_thresh = {packet_upload_loss_thresh}, packet_download_loss_thresh = {packet_download_loss_thresh}, periodicity = {periodicity}, tx_rx_pairs = {tx_rx_pairs}, tx_users = {tx_users},  RB_needed_UL = {RB_needed_UL}, RB_needed_DL = {RB_needed_DL}  and {deployment} deployment", file = open(folder_name + "/results.txt", "a"), flush = True)

        train_py_env = UAV_network(I, drones_coverage, "eval_net", folder_name, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)
        eval_py_env = UAV_network(I, drones_coverage, "eval_net", folder_name, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, T)

        train_env = tf_py_environment.TFPyEnvironment(train_py_env) # doesn't print out
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        
        train_env.reset()
        eval_env.reset()
        
        final_step_rewards = []
        dqn_returns = []
        final_step_UAV_rewards = []
        
        initial_collect_episodes = 1000  # @param {type:"integer"}  # collect_data runs this number of steps for the first time, not in REINFORCE agent
        
        batch_size = 16  # @param {type:"integer"}
        
    
        #### Agent 

        q_net = q_network.QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params)
        
        target_q_network = q_network.QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params)
        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate) # same

        train_step_counter = tf.Variable(0)


    ## https://github.com/tensorflow/agents/blob/755b43c78bb50e36b1331acc9492be599997a47f/tf_agents/agents/dqn/dqn_agent.py#L113

        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            gamma = set_gamma,
            target_q_network = target_q_network,
            target_update_tau = 1.0,
            target_update_period = 1,
            gradient_clipping = None,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
        
        # decay epsilon parameters, relates to exploration value though the param names are in terms of learning rate

        start_epsilon = 0.2
        end_learning_rate = 0.01
        decay_steps = 40_000
        
        epsilon = tf.compat.v1.train.polynomial_decay(
                                                    learning_rate = start_epsilon,
                                                    global_step = agent.train_step_counter.numpy(), # current_step
                                                    decay_steps = decay_steps,
                                                    power = 1.0,
                                                    #cycle = True,
                                                    end_learning_rate=end_learning_rate)
        
        ## tf.compat.v1.train.polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate=0.0001, power=1.0,cycle=False, name=None)
        
        # decay epsilon ends

        agent.initialize()
        
        #### Policies are properties of agents, sometimes optimizers are also properties of agents
        
        eval_policy = agent.policy
        collect_policy = agent.collect_policy
        # collect_policy._epsilon = epsilon
        
        
        tf_policy_saver = policy_saver.PolicySaver(agent.policy) 
        
        if agent.train_step_counter.numpy()==0:
            # time.sleep(10)
            # pass
            print(f"\nDQN scheduling and {deployment} placement with {I} users, coverage is {train_py_env.act_coverage}, RB_total_UL is {RB_total_UL}, RB_total_DL = {RB_total_DL}, action space size is {train_py_env.action_size} \n\n", file = open(folder_name + "/action_space.txt", "a"), flush = True)
            
            print(f"\nDQN scheduling and {deployment} placement with {I} users, coverage is {train_py_env.act_coverage}, RB_total_UL is {RB_total_UL}, RB_total_DL = {RB_total_DL}, action space size is {train_py_env.action_size} ", file = open(folder_name + "/results.txt", "a"), flush = True)
        
        if verbose:
            print(f"DQN reward discount rate = {agent._gamma}")
            print(f"\nDQN eval_policy = {eval_policy}, collect_policy = {collect_policy} with epsilon = {collect_policy._epsilon}")
            # DQN eval_policy = <tf_agents.policies.greedy_policy.GreedyPolicy object at 0x7fa02c4a2700>, collect_policy = <tf_agents.policies.epsilon_greedy_policy.EpsilonGreedyPolicy object at 0x7fa02c51a8e0> with epsilon 0.1        time.sleep(5)
        
        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec()) # used once in collect_episode first time with initial_collect_episodes

        
        time_step = train_env.reset()
        random_policy.action(time_step)


        #### Metrics and Evaluation
        def compute_avg_return(environment, policy, num_episodes=100):
            total_return = 0.0
            for i in range(num_episodes):

                time_step = environment.reset()
                episode_return = 0.0

                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    time_step = environment.step(action_step.action)
                    episode_return += time_step.reward
                total_return += episode_return
                if verbose:
                    print(f'episode={i}, step reward = {time_step.reward}, episode_return={episode_return}, total_return={total_return}', flush=True)
                
                # if comet==True: 
                #     experiment.log_metric("final_return_DQN", time_step.reward.numpy()[0], step = i) # "loss",loss_val,step=i
                final_step_rewards.append(time_step.reward.numpy()[0])
                # print(f"time_step.reward.numpy()={time_step.reward.numpy()}", flush=True)
                
            avg_return = total_return / num_episodes # avg age sum over all steps
            return avg_return.numpy()[0]
        
        compute_avg_return(eval_env, random_policy, num_eval_episodes) # to see a baseline with random policy

        #### Replay Buffer
        
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)
        
        agent.collect_data_spec
        agent.collect_data_spec._fields


        
        #### Data Collection
        
        def collect_step(environment, policy, buffer):
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            all_actions.append(action_step.action.numpy()[0])
            # print(f"all_actions = {all_actions}, type = {type(all_actions)}")
            # time.sleep(2)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            buffer.add_batch(traj)

        def collect_episode(env, policy, buffer, num_episodes): # DRL's collect_episode = collect_episode+collect_steps
            for _ in range(num_episodes):
                collect_step(env, policy, buffer)

        collect_episode(train_env, random_policy, replay_buffer, initial_collect_episodes)
            
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=batch_size, 
            num_steps=2).prefetch(3)
        
        iterator = iter(dataset)
        
        #### Training the agent
        
        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        agent.train = common.function(agent.train)

        # Reset the train step
        agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training, same as DRL
        
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        returns = [avg_return]
        final_step_UAV_rewards = [sum(eval_py_env.UAV_age.values())]
        # print(vars(eval_py_env))
        
        # print(f"final_step_UAV_rewards = {final_step_UAV_rewards} and with {eval_py_env.UAV_Age}")
        # time.sleep(15)
        
        start_time = time.time()
        for ii in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
            collect_episode(train_env, agent.collect_policy, replay_buffer, collect_episodes_per_iteration)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()

            if step % log_interval == 0 and step!=0 :
                new_time = time.time() ## seconds
                time_elapsed = np.round(new_time - start_time,2) ## seconds
                rate = np.round(time_elapsed/step, 3) # seconds/episode
                remaining_seconds = (num_iterations - step)*rate
                x = datetime.datetime.now()
                finish_time = x + datetime.timedelta(seconds=remaining_seconds)
                t = time.localtime()
                current_time = time.strftime("%H:%M:%S", t)
                print(f'step = {step}, loss = {train_loss}, Average Age = {np.mean(final_step_rewards[-5:])}', flush=True)
                print(f"current_time = {current_time}, time since start = {time_elapsed} seconds, rate = {round((1/rate),2)} eps/sec, finish_time = {finish_time} \n\n")

            if step % eval_interval == 0:
                avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
                # print('step = {0}: Average Return = {1}'.format(step, avg_return), flush =True)
                returns.append(avg_return)
                final_step_UAV_rewards.append(sum(eval_py_env.UAV_age.values()))
                # print(f"final_step_UAV_rewards = {final_step_UAV_rewards} and with {eval_py_env.UAV_age}")

            if ii == num_iterations-1:
                print(f"policy saved in {policy_dir}", file = open(folder_name + "/results.txt", "a"), flush = True)
                tf_policy_saver.save(policy_dir)
                
        dqn_returns = returns
        
        # print(f"dqn_UL_schedule = {eval_py_env.dqn_UL_schedule}, dqn_DL_schedule = {eval_py_env.dqn_DL_schedule}")
        
        pickle.dump(eval_py_env.age_dist_UAV_slot_wise, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_age_dist_UAV_slot_wise.pickle", "wb"))
        pickle.dump(eval_py_env.age_dist_dest_slot_wise, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_age_dist_dest_slot_wise.pickle", "wb"))
        
        pickle.dump(dqn_returns, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_returns.pickle", "wb"))
        pickle.dump(final_step_UAV_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_final_step_UAV_rewards.pickle", "wb"))
        pickle.dump(final_step_rewards, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_final_step_rewards.pickle", "wb"))
        
        pickle.dump(eval_py_env.tx_attempt_dest, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_tx_attempt_dest.pickle", "wb"))
        pickle.dump(eval_py_env.tx_attempt_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_tx_attempt_UAV.pickle", "wb"))

        pickle.dump(eval_py_env.sample_time, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_sample_time.pickle", "wb"))
        pickle.dump(eval_py_env.age_dist_dest, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_age_dist_dest.pickle", "wb"))
        pickle.dump(eval_py_env.age_dist_UAV, open(folder_name + "/" + deployment + "/" + str(I) + "U_DQN_age_dist_UAV.pickle", "wb"))
        
        pickle.dump(eval_py_env.attempt_upload, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_attempt_upload.pickle", "wb"))
        pickle.dump(eval_py_env.success_upload, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_success_upload.pickle", "wb"))
        pickle.dump(eval_py_env.attempt_download, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_attempt_download.pickle", "wb"))
        pickle.dump(eval_py_env.success_download, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_success_download.pickle", "wb"))
        
        pickle.dump(eval_py_env.dqn_UL_schedule, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_UL_schedule.pickle", "wb"))    
        pickle.dump(eval_py_env.dqn_DL_schedule, open(folder_name + "/" + deployment + "/" + str(I) + "U_dqn_DL_schedule.pickle", "wb"))
        
        print("\nDQN scheduling ", deployment, " placement, ", I, " users. MEAN of final_step_rewards = ", np.mean(final_step_rewards[-5:]), ". MEAN of overall_ep_reward = ", np.mean(dqn_returns[-5:]), " MIN and MAX of overall_ep_reward = ", np.min(dqn_returns),", ", np.max(dqn_returns),". Similarly for final_step_UAV_rewards - MEAN = ",{np.mean(final_step_UAV_rewards)}, ", MIN and MAX of final_step_UAV_rewards = ", np.min(final_step_UAV_rewards),", ", np.max(final_step_UAV_rewards)," end with final state of ", eval_py_env._state, " with shape ", np.shape(eval_py_env._state), ", min PDR_upload = ", {np.min(eval_py_env.PDR_upload)}, ", max PDR_upload = ", {np.max(eval_py_env.PDR_upload)}, ", min PDR_download = ", {np.min(eval_py_env.PDR_download)}, ", max PDR_upload = ", {np.max(eval_py_env.PDR_download)}, ", max_total_packet_lost_upload = ", np.max(eval_py_env.packet_lost_upload), ", min_total_packet_lost_upload = ", np.min(eval_py_env.packet_lost_upload), ", max_total_packet_lost_download = ", np.max(eval_py_env.packet_lost_download), ", min_total_packet_lost_download = ", np.min(eval_py_env.packet_lost_download), flush = True)
        
        print("\nDQN scheduling ", deployment, " placement, ", I, " users. MEAN of final_step_rewards = ", np.mean(final_step_rewards[-5:]), ". MEAN of overall_ep_reward = ", np.mean(dqn_returns[-5:]), " MIN and MAX of overall_ep_reward = ", np.min(dqn_returns),", ", np.max(dqn_returns),". Similarly for final_step_UAV_rewards - MEAN = ",{np.mean(final_step_UAV_rewards)}, ", MIN and MAX of final_step_UAV_rewards = ", np.min(final_step_UAV_rewards),", ", np.max(final_step_UAV_rewards)," end with final state of ", eval_py_env._state, " with shape ", np.shape(eval_py_env._state), ", min PDR_upload = ", {np.min(eval_py_env.PDR_upload)}, ", max PDR_upload = ", {np.max(eval_py_env.PDR_upload)}, ", min PDR_download = ", {np.min(eval_py_env.PDR_download)}, ", max PDR_upload = ", {np.max(eval_py_env.PDR_download)}, ", max_total_packet_lost_upload = ", np.max(eval_py_env.packet_lost_upload), ", min_total_packet_lost_upload = ", np.min(eval_py_env.packet_lost_upload), ", max_total_packet_lost_download = ", np.max(eval_py_env.packet_lost_download), ", min_total_packet_lost_download = ", np.min(eval_py_env.packet_lost_download), file = open(folder_name + "/results.txt", "a"), flush = True)

        
        print(f"DQN ended for {I} users and {deployment} deployment")
        return dqn_returns, final_step_rewards, np.mean(dqn_returns)
        
    elif mode == "deploy":
        
        final_step_rewards = []
        
        print(f"DQN deployment started")
        
        
        powergrid_simulation_duration = 2000 # in slots
        
        eval_py_env = UAV_network(I, drones_coverage, "eval_net", folder_name, packet_upload_loss_thresh, packet_download_loss_thresh, periodicity, adj_matrix, tx_rx_pairs, tx_users, RB_needed_UL, RB_needed_DL, BS_location, user_locations, powergrid_simulation_duration)

        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        
        model_path = "/home/biplav/AoI/results/dist_AoI_ver4/dqn_policy/policy/"
        policy = tf.saved_model.load(model_path)

        def compute_avg_return(environment, policy, num_episodes):
            total_return = 0.0
            for i in range(num_episodes):

                time_step = environment.reset()
                episode_return = 0.0

                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    time_step = environment.step(action_step.action)
                    episode_return += time_step.reward
                total_return += episode_return
                if verbose:
                    print(f'episode={i}, step reward = {time_step.reward}, episode_return={episode_return}, total_return={total_return}', flush=True)
                
                # if comet==True: 
                #     experiment.log_metric("final_return_DQN", time_step.reward.numpy()[0], step = i) # "loss",loss_val,step=i
                final_step_rewards.append(time_step.reward.numpy()[0])
                # print(f"time_step.reward.numpy()={time_step.reward.numpy()}", flush=True)
                
            avg_return = total_return / num_episodes # avg age sum over all steps
            return avg_return.numpy()[0]
        
        deploy_returns = compute_avg_return(eval_env, policy, num_episodes=10)
        
        
        print("\nDQN scheduling ", deployment, " placement, ", I, " users - MEAN of final_step_rewards = ", np.mean(final_step_rewards[-5:]), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", deploy_returns, ".  end with final state of ", eval_py_env._state, " with shape ", np.shape(eval_py_env._state))
        
        print("\nDQN scheduling ", deployment, " placement, ", I, " users - MEAN of final_step_rewards = ", np.mean(final_step_rewards[-5:]), " MIN and MAX of final_step_rewards = ", np.min(final_step_rewards),", ", np.max(final_step_rewards), " and avg of overall_ep_reward = ", deploy_returns, ".  end with final state of ", eval_py_env._state, " with shape ", np.shape(eval_py_env._state), file = open(folder_name + "/results.txt", "a"), flush = True)
        
        print(f"DQN ended for {I} users and {deployment} deployment")

        return deploy_returns, final_step_rewards, np.mean(deploy_returns)