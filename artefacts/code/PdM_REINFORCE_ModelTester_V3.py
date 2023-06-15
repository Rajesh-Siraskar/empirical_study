#!/usr/bin/env python
# coding: utf-8

# Milling Tool Wear Maintenance Policy using the REINFORCE algorithm
# V.1.0: Running model tests with pre-trained models

print (' ====== REINFORCE for Predictive Maintenance V.3.4 13-Jun-2023 ======')
print (' * Change log * SB-3 episodes set to 10 K\n')
print ('- Loading packages...')
import datetime
import os

import pickle
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DQN

from milling_tool_environment import MillingTool_SS_NT, MillingTool_MS_V3
from utilities import compute_metrics, compute_metrics_simple, write_metrics_report, store_results, plot_learning_curve, single_axes_plot, lnoise
from utilities import two_axes_plot, two_variable_plot, plot_error_bounds, test_script, write_test_results, downsample, save_model, load_model, clean_up_files
from utilities import add_performance_columns, summary_performance_metrics

from reinforce_classes import PolicyNetwork, Agent

# Auto experiment file structure
print ('- Loading pre-trained PdM models...')
df_expts = pd.read_csv('TestModels.csv')
df_expts = add_performance_columns(df_expts)

n_expts = len(df_expts.index)

experiment_summary = []

for n_expt in range(n_expts):
    dt = datetime.datetime.now()
    dt_d = dt.strftime('%d-%b-%Y')
    dt_t = dt.strftime('%H_%M_%S')
    dt_m = dt.strftime('%d-%H%M')

    # The pre-trained model to be tested
    MODEL_FILE = df_expts['model_file'][n_expt]
    print('\n', 120*'-')
    print(f' ---- [{dt_t}] Test run: {n_expt}  ---- ')
    if os.path.isfile(MODEL_FILE):
        agent_RF = load_model(MODEL_FILE)
        print(f'* Loaded model: {MODEL_FILE}')
    else:
        print(f' !!! ERROR: Unable to load model: {MODEL_FILE}')

    # Load experiment parameters
    ENVIRONMENT_CLASS = df_expts['environment'][n_expt]
    ENVIRONMENT_INFO = df_expts['environment_info'][n_expt]
    ENVIRONMENT_INFO = f'{ENVIRONMENT_INFO}-{ENVIRONMENT_CLASS}'
    DATA_FILE = df_expts['data_file'][n_expt]

    # Load model training parameters from pre-trained model object
    # Use these parameters for training the SB-3 agents
    R1 = agent_RF.model_parameters['R1']
    R2 = agent_RF.model_parameters['R2']
    R3 = agent_RF.model_parameters['R3']
    WEAR_THRESHOLD = agent_RF.model_parameters['WEAR_THRESHOLD']
    THRESHOLD_FACTOR = agent_RF.model_parameters['THRESHOLD_FACTOR']
    ADD_NOISE = agent_RF.model_parameters['ADD_NOISE']
    BREAKDOWN_CHANCE =  agent_RF.model_parameters['BREAKDOWN_CHANCE']
    EPISODES = agent_RF.model_parameters['EPISODES']
    MILLING_OPERATIONS_MAX = agent_RF.model_parameters['MILLING_OPERATIONS_MAX']
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
    VERSION = f'{n_expt}_{ver_prefix}_{l_noise}_'

    METRICS_METHOD = 'binary' # average method = {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’}

    CONSOLIDATED_METRICS_FILE = f'{RESULTS_FOLDER}/TEST_CONSOLIDATED_METRICS.csv'
    RESULTS_FILE = f'{RESULTS_FOLDER}/{VERSION}_test_results_{dt_d}_{dt_m}.csv'
    METRICS_FILE = f'{RESULTS_FOLDER}/{VERSION}_metrics.csv'
    EXPTS_REPORT = f'{RESULTS_FOLDER}/RF_performance_summary_{dt_d}_{dt_m}.csv'
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


    # 4. Down sample main data to create test set
    if TEST_SR:
        # 4. Split into train and test
        df_train = downsample(df_normalized, TRAIN_SR)
        df_train.to_csv('TempTrain.csv')
        df_train = pd.read_csv('TempTrain.csv')

        df_test = downsample(df_normalized, TEST_SR)
        df_test.to_csv('TempTest.csv')
        df_test = pd.read_csv('TempTest.csv')
        print(f'- Tool wear data split into train ({len(df_train.index)} records) and test ({len(df_test.index)} records).')
    else:
        df_train = df_normalized
        df_test = pd.read_csv(TEST_FILE)
        print(f'* Separate test data provided: {TEST_FILE} - ({len(df_test.index)} records).')

    n_records = len(df_train.index)
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
    else:
        print(' ERROR - initatizing environment')

    # ### Generate a balanced test set
    idx_replace_cases = df_test.index[df_test['ACTION_CODE'] >= 1.0]
    idx_normal_cases = df_test.index[df_test['ACTION_CODE'] < 1.0]

    # Process results
    print(f'\n- Test PRE-TRAINED REINFORCE model: {MODEL_FILE}')
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

    print('\n',120*'=')
    expt_summary = f'   Expt. {n_expt}: {ENVIRONMENT_INFO} - {l_noise} Pr: {avg_Pr:0.3f} \t Rc: {avg_Rc:0.3f} \t F1:{avg_F1:0.3f}'
    experiment_summary.append(expt_summary)
    print(expt_summary)
    print(120*'=','\n')

    print(f'- REINFORCE Test results written to file: {RESULTS_FILE}.\n')

    ## Add model training hyper parameters and save model, if metrics > 0.65
    df_expts.loc[n_expt, 'model_file_tested'] = MODEL_FILE

    # ## Stable-Baselines Algorithms
    print('* Train Stable-Baselines-3 A2C, DQN and PPO models...')

    # For stable_baselines3 algos - we run for a total timesteps of EPISODES*MILLING_OPERATIONS_MAX
    # total_timesteps = EPISODES*MILLING_OPERATIONS_MAX
    total_timesteps = 10_000

    algos = ['A2C','DQN','PPO']
    SB_agents = []
    for SB_ALGO in algos:
        if SB_ALGO.upper() == 'A2C': agent_SB = A2C('MlpPolicy', env, tensorboard_log="./tensorboard/")
        if SB_ALGO.upper() == 'DQN': agent_SB = DQN('MlpPolicy', env, tensorboard_log="./tensorboard/")
        if SB_ALGO.upper() == 'PPO': agent_SB = PPO('MlpPolicy', env, tensorboard_log="./tensorboard/")

        print(f'- Training Stable-Baselines-3 {SB_ALGO} algorithm...')
        if ENVIRONMENT_CLASS == 'SS':
            env = MillingTool_SS_NT(df_train, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)
        elif ENVIRONMENT_CLASS == 'MS':
            env = MillingTool_MS_V3(df_train, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)
        else:
            print(' ERROR - initatizing environment')

        agent_SB.learn(total_timesteps=total_timesteps)
        SB_agents.append(agent_SB)

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

    print(f'- Updating summary performance metrics.')
    summary_performance_metrics(df_expts, n_expt, algo_metrics)

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
