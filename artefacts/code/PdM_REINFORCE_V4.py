#!/usr/bin/env python
# coding: utf-8

# Milling Tool Wear Maintenance Policy using the REINFORCE algorithm
# V.3.0: Add cleaning up files. If the performance is not satisfactory -
#           delete the files, also do not train and test SB models

MIN_MODEL_PERFORMANCE = -1.0
SB3_EPISODES = 10_000

print ('\n ====== REINFORCE for Predictive Maintenance ======')
print ('        V.4.0 14-Jun-2023 Training Time\n')
print ('- Loading packages...')
from datetime import datetime
import time
import os
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DQN

from milling_tool_environment import MillingTool_SS_NT, MillingTool_MS_V3
from utilities import compute_metrics, compute_metrics_simple, write_metrics_report, store_results, plot_learning_curve, single_axes_plot, lnoise
from utilities import two_axes_plot, two_variable_plot, plot_error_bounds, test_script, write_test_results, downsample, save_model, load_model, clean_up_files
from reinforce_classes import PolicyNetwork, Agent

# Auto experiment file structure
print ('- Loading Experiments...')
df_expts = pd.read_csv('Experiments.csv')
df_expts['Pr'] = 0.0
df_expts['Rc'] = 0.0
df_expts['F1'] = 0.0
df_expts['model_file'] = 'Not satisfactory'

# Record training time
df_expts['RF_time'] = 0.0
df_expts['A2C_time'] = 0.0
df_expts['DQN_time'] = 0.0
df_expts['PPO_time'] = 0.0

n_expts = len(df_expts.index)

experiment_summary = []

for n_expt in range(n_expts):
    dt = datetime.now()
    dt_d = dt.strftime('%d-%b-%Y')
    dt_t = dt.strftime('%H_%M_%S')
    dt_m = dt.strftime('%d-%H%M')

    # Load experiment parameters
    ENVIRONMENT_CLASS = df_expts['environment'][n_expt]
    ENVIRONMENT_INFO = df_expts['environment_info'][n_expt]
    ENVIRONMENT_INFO = f'{ENVIRONMENT_INFO}-{ENVIRONMENT_CLASS}'
    DATA_FILE = df_expts['data_file'][n_expt]
    R1 = df_expts['R1'][n_expt]
    R2 = df_expts['R2'][n_expt]
    R3 = df_expts['R3'][n_expt]
    WEAR_THRESHOLD = df_expts['wear_threshold'][n_expt]
    THRESHOLD_FACTOR = df_expts['threshold_factor'][n_expt]
    ADD_NOISE = df_expts['add_noise'][n_expt]
    BREAKDOWN_CHANCE = df_expts['breakdown_chance'][n_expt]
    EPISODES = df_expts['episodes'][n_expt]
    MILLING_OPERATIONS_MAX = df_expts['milling_operations_max'][n_expt]
    ver_prefix = df_expts['version_prefix'][n_expt]
    TEST_INFO = df_expts['test_info'][n_expt]
    TEST_CASES = df_expts['test_cases'][n_expt]
    TEST_ROUNDS = df_expts['test_rounds'][n_expt]
    RESULTS_FOLDER = df_expts['results_folder'][n_expt]

    TEST_FILE = df_expts['test_file'][n_expt]
    TRAIN_SR = df_expts['train_sample_rate'][n_expt]
    TEST_SR = df_expts['test_sample_rate'][n_expt]

    ## Read data
    df = pd.read_csv(DATA_FILE)
    n_records = len(df.index)
    l_noise = lnoise(ADD_NOISE, BREAKDOWN_CHANCE)
    #VERSION = f'{ver_prefix}_{l_noise}_{WEAR_THRESHOLD}_{THRESHOLD_FACTOR}_{R3}_{EPISODES}_{MILLING_OPERATIONS_MAX}_'
    VERSION = f'{n_expt}_{ver_prefix}_{l_noise}_'
    print('\n', 120*'-')
    print(f'\n [{dt_t}] Experiment {n_expt}: {VERSION}')

    model_file = f'models/RF_Model_{n_expt}_{ver_prefix}_{l_noise}_{dt_m}.mdl'

    METRICS_METHOD = 'binary' # average method = {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’}
    WEAR_THRESHOLD_NORMALIZED = 0.0 # normalized to the max wear threshold

    # Policy network learning parameters
    gamma = 0.99
    alpha = 0.01

    CONSOLIDATED_METRICS_FILE = f'{RESULTS_FOLDER}/TEST_CONSOLIDATED_METRICS.csv'
    RESULTS_FILE = f'{RESULTS_FOLDER}/{VERSION}_test_results_{dt_d}_{dt_m}.csv'
    METRICS_FILE = f'{RESULTS_FOLDER}/{VERSION}_metrics.csv'
    EXPTS_REPORT = f'{RESULTS_FOLDER}/Experiment_Results_{dt_d}_{dt_m}.csv'

    print('\n- Columns added to results file: ', RESULTS_FILE)
    results = ['Date', 'Time', 'Round', 'Environment', 'Training_data', 'Wear_Threshold', 'Test_data', 'Algorithm', 'Episodes', 'Normal_cases', 'Normal_error',
               'Replace_cases', 'Replace_error', 'Overall_error',
               'Precision', 'Recall', 'F_Beta_0_5', 'F_Beta_0_75', 'F_1_Score']
    write_test_results(results, RESULTS_FILE)


    # ## Data pre-process
    # 1. Add noise
    # 2. Add ACTION_CODE based on tool wear threshold
    # 3. Normalize data base
    # 4. Split into train and test

    # 1. Add noise
    if ADD_NOISE:
        df['tool_wear'] += np.random.normal(0, 1, n_records)/ADD_NOISE

    # 2. Add ACTION code
    df['ACTION_CODE'] = np.where(df['tool_wear'] < WEAR_THRESHOLD, 0.0, 1.0)

    # 3. Normalize
    WEAR_MIN = df['tool_wear'].min()
    WEAR_MAX = df['tool_wear'].max()
    WEAR_THRESHOLD_ORG_NORMALIZED = (WEAR_THRESHOLD-WEAR_MIN)/(WEAR_MAX-WEAR_MIN)
    WEAR_THRESHOLD_NORMALIZED = THRESHOLD_FACTOR*(WEAR_THRESHOLD-WEAR_MIN)/(WEAR_MAX-WEAR_MIN)
    df_normalized = (df-df.min())/(df.max()-df.min())
    df_normalized['ACTION_CODE'] = df['ACTION_CODE']
    print(f'- Tool wear data imported ({len(df.index)} records).')

    # 4. Test file -or- create test file
    if TRAIN_SR:
        # 4. Split into train and test
        df_train = downsample(df_normalized, TRAIN_SR)
        df_train.to_csv('TempTrain.csv')
        df_train = pd.read_csv('TempTrain.csv')

        df_test = downsample(df_normalized, TEST_SR)
        df_test.to_csv('TempTest.csv')
        df_test = pd.read_csv('TempTest.csv')
        print(f'- Tool wear data split into train ({len(df_train.index)} records) and test ({len(df_test.index)} records).')
    else:
        # 4. Split into train and test
        df_train = df_normalized
        df_test = pd.read_csv(TEST_FILE)
        print(f'* Separate test data provided: {TEST_FILE} - ({len(df_test.index)} records).')

    n_records = len(df_train.index)
    x = [n for n in range(n_records)]
    y1 = df_train['tool_wear']
    y2 = df_train['ACTION_CODE']
    wear_plot = f'{RESULTS_FOLDER}/{VERSION}_wear_plot.png'
    title=f'Tool Wear (mm) data\n{VERSION}'
    two_axes_plot(x, y1, y2, title=title, x_label='Time', y1_label='Tool Wear (mm)', y2_label='Action code (1=Replace)', xticks=20, file=wear_plot, threshold_org = WEAR_THRESHOLD_ORG_NORMALIZED, threshold=WEAR_THRESHOLD_NORMALIZED)


    # ## Milling Tool Environment -
    # 1. MillingTool_SS: Single state: tool_wear and time
    # 2. MillingTool_MS: Multie-state: force_x; force_y; force_z; vibration_x; vibration_y; vibration_z; acoustic_emission_rms; tool_wear
    # - Note: ACTION_CODE is only used for evaluation later (testing phase) and is NOT passed as part of the environment states

    if ENVIRONMENT_CLASS == 'SS':
        env = MillingTool_SS_NT(df_train, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)
        env_test = MillingTool_SS_NT(df_test, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)
    elif ENVIRONMENT_CLASS == 'MS':
        env = MillingTool_MS_V3(df_train, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)
        env_test = MillingTool_MS_V3(df_test, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)
    elif ENVIRONMENT_CLASS == 'MS-V4':
        env = MillingTool_MS_V4(df_train, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)
        env_test = MillingTool_MS_V4(df_test, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)
    else:
        print(' ERROR - initatizing environment')

    # ## REINFORCE RL Algorithm
    ### Main loop
    print('\n* Train REINFORCE model...')
    rewards_history = []
    loss_history = []
    training_stats = []

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent_RF = Agent(input_dim, output_dim, alpha, gamma)

    time_RF = time.time()
    for episode in range(EPISODES):
        state = env.reset()

        # Sample a trajectory
        for t in range(MILLING_OPERATIONS_MAX): # Max. milling operations desired
            action = agent_RF.act(state)
            state, reward, done, info = env.step(action)
            agent_RF.rewards.append(reward)
            #env.render()
            if done:
                # print('** DONE **', info)
                break

        # Learn during this episode
        loss = agent_RF.learn() # train per episode
        total_reward = sum(agent_RF.rewards)

        # Record statistics for this episode
        rewards_history.append(total_reward)
        loss_history.append(loss.item()) # Extract values from list of torch items for plotting

        # On-policy - so discard all data
        agent_RF.onpolicy_reset()

        if (episode%100 == 0):
            # print(f'[{episode:04d}] Loss: {loss:>10.2f} | Reward: {total_reward:>10.2f} | Ep.length: {env.ep_length:04d}')
            print(f'[{episode:04d}] Loss: {loss:>10.2e} | Reward: {total_reward:>10.2e} | Ep.length: {env.ep_length:04d}')

    # end RF training
    time_RF = time.time() - time_RF

    x = [i for i in range(EPISODES)]

    ## Moving average for rewards
    ma_window_size = 10
    # # Convert error array to pandas series
    rewards = pd.Series(rewards_history)
    windows = rewards.rolling(ma_window_size)
    moving_avg = windows.mean()
    moving_avg_lst = moving_avg.tolist()
    y1 = rewards
    y2 = moving_avg_lst

    filename = f'{RESULTS_FOLDER}/{VERSION}_Avg_episode_rewards.png'
    two_variable_plot(x, y1, y2, 'Avg. rewards per episode', VERSION, 'Episode', 'Avg. Rewards', 'Moving Avg.', 50, filename)

    # plot_error_bounds(x, y1)

    filename = f'{RESULTS_FOLDER}/{VERSION}_Episode_Length.png'
    single_axes_plot(x, env.ep_length_history, 'Episode length', VERSION, 'Episode', 'No of milling operations', 50, 0.0, filename)

    filename = f'{RESULTS_FOLDER}/{VERSION}_Tool_Replacements.png'
    single_axes_plot(x, env.ep_tool_replaced_history, 'Tool replacements per episode', VERSION, 'Episode', 'Replacements', 50, 0.0, filename)

    # ### Generate a balanced test set
    idx_replace_cases = df_test.index[df_test['ACTION_CODE'] >= 1.0]
    idx_normal_cases = df_test.index[df_test['ACTION_CODE'] < 1.0]

    # Process results
    # eps = [i for i in range(EPISODES)]
    # store_results(RF_TRAINING_FILE, training_round, eps, rewards_history, env.ep_tool_replaced_history)
    print('\n- Test REINFORCE model...')
    # print(80*'-')
    # print(f'Algorithm\tNormal\terr.%\tReplace\terr.%\tOverall err.%')
    # print(80*'-')
    avg_Pr = avg_Rc = avg_F1 = 0.0

    for test_round in range(TEST_ROUNDS):
        # Create test cases
        idx_replace_cases = np.random.choice(idx_replace_cases, int(TEST_CASES/2), replace=False)
        idx_normal_cases = np.random.choice(idx_normal_cases, int(TEST_CASES/2), replace=False)
        test_cases = [*idx_normal_cases, *idx_replace_cases]

        results = test_script(METRICS_METHOD, test_round, df_test, 'REINFORCE', EPISODES, env_test, ENVIRONMENT_INFO, agent_RF,
                              test_cases, TEST_INFO, DATA_FILE, WEAR_THRESHOLD, RESULTS_FILE)
        write_test_results(results, RESULTS_FILE)
        avg_Pr += results[14]
        avg_Rc += results[15]
        avg_F1 += results[16]

    avg_Pr = avg_Pr/TEST_ROUNDS
    avg_Rc = avg_Rc/TEST_ROUNDS
    avg_F1 = avg_F1/TEST_ROUNDS

    df_expts.loc[n_expt, 'Pr'] = avg_Pr
    df_expts.loc[n_expt, 'Rc'] = avg_Rc
    df_expts.loc[n_expt, 'F1'] = avg_F1
    df_expts.loc[n_expt, 'RF_time'] = time_RF

    expt_summary = f'Expt. {n_expt}: {ENVIRONMENT_INFO} - {l_noise} Pr: {avg_Pr:0.3f} \t Rc: {avg_Rc:0.3f} \t F1:{avg_F1:0.3f}'
    experiment_summary.append(expt_summary)
    print(expt_summary)

    print(f'- REINFORCE Test results written to file: {RESULTS_FILE}.\n')

    ## Add model training hyper parameters and save model, if metrics > 0.65
    if avg_Pr > MIN_MODEL_PERFORMANCE and avg_Rc > MIN_MODEL_PERFORMANCE and avg_F1 > MIN_MODEL_PERFORMANCE:
        print(f'\n*** REINFORCE model performance satisfactory. Saving model to {model_file} ***\n')
        agent_RF.model_parameters = {'R1':R1, 'R2':R2, 'R3':R3, 'WEAR_THRESHOLD':WEAR_THRESHOLD, 'THRESHOLD_FACTOR':THRESHOLD_FACTOR, 'ADD_NOISE':ADD_NOISE, 'BREAKDOWN_CHANCE':BREAKDOWN_CHANCE, 'EPISODES':EPISODES, 'MILLING_OPERATIONS_MAX':MILLING_OPERATIONS_MAX}
        save_model(agent_RF, model_file)
        df_expts.loc[n_expt, 'model_file'] = model_file

        # ## Stable-Baselines Algorithms
        print('* Train Stable-Baselines-3 A2C, DQN and PPO models...')

        total_timesteps = SB3_EPISODES
        algos = ['A2C','DQN','PPO']
        SB_agents = []

        for SB_ALGO in algos:
            if SB_ALGO.upper() == 'A2C': agent_SB = A2C('MlpPolicy', env)
            if SB_ALGO.upper() == 'DQN': agent_SB = DQN('MlpPolicy', env)
            if SB_ALGO.upper() == 'PPO': agent_SB = PPO('MlpPolicy', env)

            print(f'- Training Stable-Baselines-3 {SB_ALGO} algorithm...')
            time_SB = time.time()
            agent_SB.learn(total_timesteps=total_timesteps)
            time_SB = time.time() - time_SB
            time_column = f'{SB_ALGO.upper()}_time'
            df_expts.loc[n_expt, time_column] = time_SB

            SB_agents.append(agent_SB)
            # print(f'- Save Stable-Baselines-3 model')
            # model_file_SB = f'models/{SB_ALGO}_{n_expt}_{ver_prefix}_{l_noise}_{dt_m}.mdl'
            # save_model(agent_SB, model_file_SB)

        n = 0
        for agent_SB in SB_agents:
            print(f'- Testing Stable-Baselines-3 {agent_SB} model...')
            # print(80*'-')
            # print(f'Algo.\tNormal\tErr.%\tReplace\tErr.%\tOverall err.%')
            # print(80*'-')
            for test_round in range(TEST_ROUNDS):
                # Create test cases
                idx_replace_cases = np.random.choice(idx_replace_cases, int(TEST_CASES/2), replace=False)
                idx_normal_cases = np.random.choice(idx_normal_cases, int(TEST_CASES/2), replace=False)
                test_cases = [*idx_normal_cases, *idx_replace_cases]
                results = test_script(METRICS_METHOD, test_round, df_test, algos[n], EPISODES, env_test, ENVIRONMENT_INFO,
                                      agent_SB, test_cases, TEST_INFO, DATA_FILE, WEAR_THRESHOLD, RESULTS_FILE)
                write_test_results(results, RESULTS_FILE)
                # end test loop

            n += 1
            # end SB agents loop


        ### Create a consolidated algorithm wise metrics summary

        print(f'* Test Report: Algorithm level consolidated metrics will be written to: {METRICS_FILE}.')

        header_columns = [VERSION]
        write_test_results(header_columns, METRICS_FILE)
        header_columns = ['Date', 'Time', 'Environment', 'Noise', 'Breakdown_chance', 'Train_data', 'env.R1', 'env.R2', 'env.R3', 'Wear threshold', 'Look-ahead Factor', 'Episodes', 'Terminate on', 'Test_info', 'Test_cases', 'Metrics_method', 'Version']
        write_test_results(header_columns, METRICS_FILE)

        dt_t = dt.strftime('%H:%M:%S')
        noise_info = 'None' if ADD_NOISE == 0 else (1/ADD_NOISE)
        header_info = [dt_d, dt_t, ENVIRONMENT_INFO, noise_info, BREAKDOWN_CHANCE, DATA_FILE, env.R1, env.R2, env.R3, WEAR_THRESHOLD, THRESHOLD_FACTOR, EPISODES, MILLING_OPERATIONS_MAX, TEST_INFO, TEST_CASES, METRICS_METHOD, VERSION]
        write_test_results(header_info, METRICS_FILE)
        write_test_results([], METRICS_FILE) # leave a blank line
        print('- Experiment related meta info written.')

        df_algo_results = pd.read_csv(RESULTS_FILE)
        # algo_metrics = compute_metrics_simple(df_algo_results)
        algo_metrics = compute_metrics(df_algo_results)

        write_metrics_report(algo_metrics, METRICS_FILE, 4)
        write_test_results([], METRICS_FILE) # leave a blank line
        print('- Algorithm level consolidated metrics reported to file.')

        write_test_results(header_columns, CONSOLIDATED_METRICS_FILE)
        write_test_results(header_info, CONSOLIDATED_METRICS_FILE)
        write_test_results([], CONSOLIDATED_METRICS_FILE) # leave a blank line
        write_metrics_report(algo_metrics, CONSOLIDATED_METRICS_FILE, 4)
        write_test_results([120*'-'], CONSOLIDATED_METRICS_FILE) # leave a blank line
        print(f'- {CONSOLIDATED_METRICS_FILE} file updated.')
        print(algo_metrics.round(3))
        print(f'- End Experiment {n_expt}')
    else:
        clean_up_files(RESULTS_FOLDER, VERSION, dt_d, dt_m)

# end for all experiments

df_expts.to_csv(EXPTS_REPORT)
print(120*'-')
print('SUMMARY REPORT')
print(120*'-')
for e in experiment_summary:
    print(e)
print(120*'=')

if os.path.isfile('TempTrain.csv'):
    os.remove('TempTrain.csv')
if os.path.isfile('TempTest.csv'):
    os.remove('TempTest.csv')
