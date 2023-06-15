# Setting background color
# ax.set_facecolor('#EFEFEF')

import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import datetime
import pickle

def add_performance_columns(df_expts):
    df_expts['RF_Pr'] = 0.0
    df_expts['RF_Pr_sd'] = 0.0
    df_expts['RF_Rc'] = 0.0
    df_expts['RF_Rc_sd'] = 0.0
    df_expts['RF_F1'] = 0.0
    df_expts['RF_F1_sd'] = 0.0

    df_expts['A2C_Pr'] = 0.0
    df_expts['A2C_Pr_sd'] = 0.0
    df_expts['A2C_Rc'] = 0.0
    df_expts['A2C_Rc_sd'] = 0.0
    df_expts['A2C_F1'] = 0.0
    df_expts['A2C_F1_sd'] = 0.0

    df_expts['DQN_Pr'] = 0.0
    df_expts['DQN_Pr_sd'] = 0.0
    df_expts['DQN_Rc'] = 0.0
    df_expts['DQN_Rc_sd'] = 0.0
    df_expts['DQN_F1'] = 0.0
    df_expts['DQN_F1_sd'] = 0.0

    df_expts['PPO_Pr'] = 0.0
    df_expts['PPO_Pr_sd'] = 0.0
    df_expts['PPO_Rc'] = 0.0
    df_expts['PPO_Rc_sd'] = 0.0
    df_expts['PPO_F1'] = 0.0
    df_expts['PPO_F1_sd'] = 0.0

    return(df_expts)

def summary_performance_metrics(df_expts, n_expt, algo_metrics):
    df_expts.loc[n_expt, 'RF_Pr'] = algo_metrics['Precision']['mean']['REINFORCE']
    df_expts.loc[n_expt, 'RF_Pr_sd'] = algo_metrics['Precision']['std']['REINFORCE']
    df_expts.loc[n_expt, 'RF_Rc'] = algo_metrics['Recall']['mean']['REINFORCE']
    df_expts.loc[n_expt, 'RF_Rc_sd'] = algo_metrics['Recall']['std']['REINFORCE']
    df_expts.loc[n_expt, 'RF_F1'] = algo_metrics['F_1_Score']['mean']['REINFORCE']
    df_expts.loc[n_expt, 'RF_F1_sd'] = algo_metrics['F_1_Score']['std']['REINFORCE']

    df_expts.loc[n_expt, 'A2C_Pr'] = algo_metrics['Precision']['mean']['A2C']
    df_expts.loc[n_expt, 'A2C_Pr_sd'] = algo_metrics['Precision']['std']['A2C']
    df_expts.loc[n_expt, 'A2C_Rc'] = algo_metrics['Recall']['mean']['A2C']
    df_expts.loc[n_expt, 'A2C_Rc_sd'] = algo_metrics['Recall']['std']['A2C']
    df_expts.loc[n_expt, 'A2C_F1'] = algo_metrics['F_1_Score']['mean']['A2C']
    df_expts.loc[n_expt, 'A2C_F1_sd'] = algo_metrics['F_1_Score']['std']['A2C']

    df_expts.loc[n_expt, 'DQN_Pr'] = algo_metrics['Precision']['mean']['DQN']
    df_expts.loc[n_expt, 'DQN_Pr_sd'] = algo_metrics['Precision']['std']['DQN']
    df_expts.loc[n_expt, 'DQN_Rc'] = algo_metrics['Recall']['mean']['DQN']
    df_expts.loc[n_expt, 'DQN_Rc_sd'] = algo_metrics['Recall']['std']['DQN']
    df_expts.loc[n_expt, 'DQN_F1'] = algo_metrics['F_1_Score']['mean']['DQN']
    df_expts.loc[n_expt, 'DQN_F1_sd'] = algo_metrics['F_1_Score']['std']['DQN']

    df_expts.loc[n_expt, 'PPO_Pr'] = algo_metrics['Precision']['mean']['PPO']
    df_expts.loc[n_expt, 'PPO_Pr_sd'] = algo_metrics['Precision']['std']['PPO']
    df_expts.loc[n_expt, 'PPO_Rc'] = algo_metrics['Recall']['mean']['PPO']
    df_expts.loc[n_expt, 'PPO_Rc_sd'] = algo_metrics['Recall']['std']['PPO']
    df_expts.loc[n_expt, 'PPO_F1'] = algo_metrics['F_1_Score']['mean']['PPO']
    df_expts.loc[n_expt, 'PPO_F1_sd'] = algo_metrics['F_1_Score']['std']['PPO']

def clean_up_files(results_folder, version, dt_d, dt_m):
    remove_files = []

    remove_files.append(f'{results_folder}/{version}_test_results_{dt_d}_{dt_m}.csv')
    remove_files.append(f'{results_folder}/{version}_metrics.csv')
    remove_files.append(f'{results_folder}/{version}_wear_plot.png')
    remove_files.append(f'{results_folder}/{version}_Avg_episode_rewards.png')
    remove_files.append(f'{results_folder}/{version}_Episode_Length.png')
    remove_files.append(f'{results_folder}/{version}_Tool_Replacements.png')

    for filename in remove_files:
        if os.path.isfile(filename):
            os.remove(filename)
            print(f'--- Cleaning: {filename} deleted!')

    print('--- Model performance not satisfactory. Files removed.')

def save_model(model, filepath):
    pickle.dump(model, open(filepath, 'wb'))
    print(f'* Model saved to {filepath}.')

def load_model(filepath):
    print(f'- Loading saved model from {filepath}.')
    return (pickle.load(open(filepath, 'rb')))

def lnoise(add_noise, breakdown_chance):
    if add_noise == 1e4 and breakdown_chance == 0.05:
        l = 'ELowNBD'
    elif add_noise == 4e3 and breakdown_chance == 0.05:
        l = 'VLowNBD'
    elif add_noise == 1e3 and breakdown_chance == 0.05:
        l = 'LowNBD'
    elif add_noise == 1e2  and breakdown_chance == 0.10:
        l = 'HighNBD'
    elif add_noise <= 0 and breakdown_chance == 0:
        l = 'NoNBD'
    else:
        l = 'ArbNBD'
    return (l)


def downsample(df, sample_rate):
    # import pandas as pd
    # df = pd.read_csv(file)
    df_downsampled = df.iloc[::sample_rate, :]
    print(f'- Down-sampling. Input data records: {len(df.index)}. Sampling rate: {sample_rate}. Expected rows {round(len(df.index)/sample_rate)}. Down-sampled to {len(df_downsampled.index)} rows.')
    return(df_downsampled)

def compute_metrics_all(df):
    metrics = df.groupby(['Algorithm']).agg({'Precision': ['mean','std'], 'Recall': ['mean','std'],
                                             'F_Beta_0_5': ['mean','std'], 'F_Beta_0_75': ['mean','std'], 'F_1_Score': ['mean','std'],
                                             'Normal_cases': ['mean'], 'Normal_error': ['mean','std'],
                                             'Replace_cases': ['mean'], 'Replace_error': ['mean','std'],
                                             'Overall_error': ['mean','std']})
    return(metrics)

def compute_metrics(df):
    metrics = df.groupby(['Algorithm']).agg({'Precision': ['mean','std'], 'Recall': ['mean','std'],
                                             'F_Beta_0_5': ['mean','std'], 'F_Beta_0_75': ['mean','std'], 'F_1_Score': ['mean','std'],
                                             'Normal_error': ['mean'], 'Replace_error': ['mean'], 'Overall_error': ['mean']})
    return(metrics)

def compute_metrics_simple(df):
    metrics = df.groupby(['Algorithm']).agg({'Precision': ['mean'], 'Recall': ['mean'],
                                             'F_Beta_0_5': ['mean'], 'F_Beta_0_75': ['mean'], 'F_1_Score': ['mean'],
                                             'Normal_error': ['mean'],'Replace_error': ['mean'], 'Overall_error': ['mean']})
    return(metrics)


def write_metrics_report(metrics, report_file, round_decimals=8):
    from pathlib import Path
    report_file = Path(report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    metrics = metrics.round(round_decimals)
    metrics.to_csv(report_file, mode='a')

def store_results(file, rounds, episodes, rewards_history, ep_tool_replaced_history):
    dt = datetime.datetime.now()
    dt_d = dt.strftime('%d-%b-%Y')
    dt_t = dt.strftime('%H:%M:%S')
    df = pd.DataFrame({'Date': dt_d, 'Time': dt_t, 'Round': rounds, 'Episode': episodes, 'Rewards': rewards_history, 'Tool_replaced': ep_tool_replaced_history})
    # Append to existing training records file
    df.to_csv(file, mode='a', index=False, header=False)
    print(f'REINFORCE algorithm results saved to {file}')

def write_test_results(results, results_file):
    from csv import writer
    with open(results_file, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(results)
        f_object.close()

    # print(f'- Test results written to file: {results_file}')

def test_script(method, training_round, df, algo, episodes, env, env_info, agent, test_cases, training_info, data_file, wear_threshold, results_file):
    dt = datetime.datetime.now()
    dt_d = dt.strftime('%d-%b-%Y')
    dt_t = dt.strftime('%H:%M:%S')

    lst_action_actual = []
    lst_action_pred = []

    cumm_error_0 = 0.0
    cumm_error_1 = 0.0
    n_0 = 0
    n_1 = 0
    for idx in test_cases:
        state = env._get_observation(idx)
        action_pred, next_state = agent.predict(state)
        action_actual = int(df['ACTION_CODE'][idx])

        lst_action_actual.append(action_actual)
        lst_action_pred.append(action_pred)

        e = int(action_pred - action_actual)
        if action_actual:
            cumm_error_1 += abs(e)
            n_1 += 1
            # print(f'    {idx:4d}: VB (mm): {state[1]*17.855:6.3f} \t Action predicted: {action_pred} \t actual: {action_actual} \t error: {e}')
        else:
            cumm_error_0 += abs(e)
            n_0 += 1
            # print(f' ** {idx:4d}: VB (mm): {state[1]*17.855:6.3f} \t Action predicted: {action_pred} \t actual: {action_actual} \t error: {e}')
    # end for. List of y_true (lst_action_actual) and y_pred (lst_action_pred) collected.
    # Proceed to compute precision recall

    # sklearn.metrics.fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')[source]

    # Compute F_Beta at 0.5, 0.75, 1.0
    (pr, rc, f1_0_5, support) = precision_recall_fscore_support(y_true=lst_action_actual, y_pred=lst_action_pred, beta=0.5, average=method, zero_division=0)
    (pr, rc, f1_0_75, support) = precision_recall_fscore_support(y_true=lst_action_actual, y_pred=lst_action_pred, beta=0.75, average=method, zero_division=0)
    (pr, rc, f1_1_0, support) = precision_recall_fscore_support(y_true=lst_action_actual, y_pred=lst_action_pred, beta=1.0, average=method, zero_division=0)

    if n_0 == 0: n_0 = 1
    if n_1 == 0: n_1 = 1

    # print(f'{algo}\t{n_0}\t{cumm_error_0/n_0:5.3f}\t{n_1}\t{cumm_error_1/n_1:5.3f}\t{(cumm_error_0 + cumm_error_1)/(n_0+n_1):5.3f}')

    # File format
    # Date	Time	Enviroment	Data_file	Test_set	Algo.	Episodes	Normal_cases	Normal_Error\
    # Replacement_cases	Replacement_Error	Overall_Error	Parameter	Value
    # 5/7/2023	51:35.2	Simple ME	Tool_Wear_VB.csv	Sampled from training	A2C	300	37	54.1%	63	54.0%	54.0%
    # Pr [14], Rc [15], F1 [18]
    results = [dt_d, dt_t, training_round, env_info, data_file, wear_threshold, training_info, algo, episodes,
               n_0, cumm_error_0/n_0, n_1, cumm_error_1/n_1, (cumm_error_0 + cumm_error_1)/(n_0+n_1),
               pr, rc, f1_0_5, f1_0_75, f1_1_0]

    return results



def plot_learning_curve(x, rewards_history, loss_history, moving_avg_n, filename):
    fig = plt.figure()
    plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, rewards_history, color='C0')
    ax.set_xlabel('Training steps', color='C0')
    ax.set_ylabel('Rewards', color='C0')
    ax.tick_params(axis='x', color='C0')
    ax.tick_params(axis='y', color='C0')

    N = len(rewards_history)
    moving_avg = np.empty(N)
    for t in range(N):
        moving_avg[t] = np.mean(loss_history[max(0, t-moving_avg_n):(t+1)])

    ax2.scatter(x, moving_avg, color='C1', s=1)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Loss', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')
    plt.savefig(filename)
    ### plt.show()

def single_axes_plot(x, y, title='', subtitle='', x_label='', y_label='', xticks=0, threshold=0.0, filename='plot.png'):

    # Plot y
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi= 80)
    ax.plot(x, y, color='tab:blue', linewidth=2)

    # Decorations
    ax.set_xlabel(x_label, fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.set_ylabel(y_label, color='tab:blue', fontsize=16)
    ax.tick_params(axis='y', rotation=0, labelcolor='tab:blue')
    ax.grid(alpha=.4)
    if threshold > 0.0:
        ax.axhline(y = threshold, color = 'r', linestyle = '--',  linewidth=1.0)
        ax.grid(alpha=.4)
    ax.set_xticks(np.arange(0, len(x), xticks))
    ax.set_xticklabels(x[::xticks], rotation=90, fontdict={'fontsize':10})
    plt.suptitle(title, fontsize=16, fontweight="bold", x=0.02, y=0.96, ha="left")
    plt.title(subtitle, fontsize=10, pad=10, loc="left")
    # ax.set_title(title, fontsize=18)
    # plt.title(subtitle)
    fig.tight_layout()
    plt.savefig(filename)
    ### plt.show()

def two_variable_plot(x, y1, y2, title='', subtitle='', x_label='', y1_label='', y2_label='', xticks=0, filename='plot.png'):
    # Plot Line1 (Left Y Axis)
    fig, ax = plt.subplots(1,1,figsize=(10, 4), dpi= 80)
    ax.plot(x, y1, color='tab:green', alpha=0.7, linewidth=0.5)
    ax.plot(x, y2, color='tab:blue', linewidth=2.0)

    # Decorations
    ax.set_xlabel(x_label, fontsize=16)
    ax.tick_params(axis='x', rotation=0, labelsize=10)
    ax.set_ylabel(y2_label, color='tab:blue', fontsize=16)
    ax.tick_params(axis='y', rotation=0, labelcolor='tab:blue')
    ax.grid(alpha=.4)
    ax.set_xticks(np.arange(0, len(x), xticks))
    ax.set_xticklabels(x[::xticks], rotation=90, fontdict={'fontsize':10})
    plt.suptitle(title, fontsize=16, fontweight="bold", x=0.02, y=0.96, ha="left")
    plt.title(subtitle, fontsize=10, pad=10, loc="left")
    # ax.set_title(title, fontsize=18)
    ax.legend(['Rewards', 'Moving avg.'])

    fig.tight_layout()
    plt.savefig(filename)
    ### plt.show()

def two_axes_plot(x, y1, y2, title='', subtitle='', x_label='', y1_label='', y2_label='', xticks=0, file='Wear_Plot.png', threshold_org=0.0, threshold=0.0,):
    # Plot Line1 (Left Y Axis)
    fig, ax1 = plt.subplots(1,1,figsize=(10, 4), dpi= 80)
    ax1.plot(x, y1, color='tab:orange', linewidth=2)

    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y2, color='tab:blue', alpha=0.5)

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel(x_label, fontsize=16)
    ax1.tick_params(axis='x', rotation=0, labelsize=10)
    ax1.set_ylabel(y1_label, color='tab:red', fontsize=16)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red')
    ax1.grid(alpha=.4)
    if threshold > 0.0:
        ax1.axhline(y = threshold_org, color = 'r', linestyle = 'dotted',  linewidth=1.0)
        ax1.grid(alpha=.3)
        ax1.axhline(y = threshold, color = 'r', linestyle = 'dashed',  linewidth=1.0)
        ax1.grid(alpha=.4)

    # ax2 (right Y axis)
    ax2.set_ylabel(y2_label, color='tab:blue', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_xticks(np.arange(0, len(x), xticks))
    ax2.set_xticklabels(x[::xticks], rotation=90, fontdict={'fontsize':10})
    # plt.suptitle(title, fontsize=16, fontweight="bold", x=0.02, y=0.96, ha="left")
    # plt.title(subtitle, fontsize=10, pad=10, loc="left")
    ax2.set_title(title, fontsize=18)
    fig.tight_layout()
    plt.savefig(file)
    ### plt.show()

def plot_error_bounds(x, y):
    import seaborn as sns
    sns.set()

    # Compute standard error
    sem = np.std(y, ddof=1) / np.sqrt(np.size(y))
    sd = np.std(y)

    plt.figure(figsize=(9, 4))
    center_line = plt.plot(x, y, 'b-')
    fill = plt.fill_between(x, y-sd, y+sd, color='b', alpha=0.2)
    plt.margins(x=0)
    plt.legend(['Rewards'])
    ### plt.show()

# # DOWN SAMPLE HIGH RES DATA
# WEAR_THRESHOLD = 0.1
# DATA_FILE = 'data\PHM_C04_MultiStateEnv.csv'
# SAVE_TO = 'data\PHM_C04_MultiStateEnv_DS.csv'
# df = pd.read_csv(DATA_FILE)
# df_downsampled = downsample(DATA_FILE, 10)
# df_downsampled.to_csv(SAVE_TO, index=False)

# # Visualize the data
# n_records = len(df_downsampled.index)
# x = [n for n in range(n_records)]
# y1 = df_downsampled['tool_wear'].values.tolist()
# y2 = df_downsampled['ACTION_CODE'].values.tolist()
# two_axes_plot(x, y1, y2, title='Tool Wear (mm) data', x_label='Time', y1_label='Tool Wear (mm)', y2_label='Action code (1=Replace)', xticks=20, threshold=WEAR_THRESHOLD)
