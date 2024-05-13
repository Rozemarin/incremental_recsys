from tqdm import tqdm
# %matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.names import *
import copy

# %config InlineBackend.figure_format = 'retina'
# plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.format'] = 'pdf'
sns.set_style('darkgrid')

def b():
    return

def visualize_time(algorithm_metrics_dict, split_data):
    (
        r_periods,
        r_total_users_items,
        r_users_ids,
        r_A_act,
        r_users_actual_watched,
        r_clean_next_portion,
        r_dirt,
        r_n_virtual_users,
        r_n_real_active,
        r_n_pure_cold,
        training,
        holdout
    ) = split_data

    algorithm_names = list(algorithm_metrics_dict.keys())

    width = .2 # width of a bar

    fig, ax_bar = plt.subplots(1, 1, figsize=(6 * 1, 5))

    real_active = 'number of real active users'
    virtual = 'number of created virtual users'

    info_dict = {
        'total_users' : [nusers for nusers, nitems in r_total_users_items[1:]],
        'total_movies' : [nitems for nusers, nitems in r_total_users_items[1:]]
    }

    for alg_idx, algorithm_name in enumerate(ALL_ALG_NAMES):
        if algorithm_name in algorithm_names:
            info_dict[algorithm_name] = algorithm_metrics_dict[algorithm_name][-1][1:-1]

    m1_t = pd.DataFrame(info_dict)

    user_movie_colors = ['tab:olive', 'tab:pink']
    real_virtual = ['tab:green', 'tab:gray']

    m1_t[['total_users','total_movies']].plot(ax=ax_bar, kind='bar', color=user_movie_colors, secondary_y=True, legend=None, alpha=0.5, width=width)

    for alg_idx, algorithm_name in enumerate(ALL_ALG_NAMES):
        if algorithm_name not in algorithm_names:
            continue
        if algorithm_name == ALL_RANDOM or algorithm_name == MOST_POPULAR:
            continue
        elif algorithm_name == SVD:
            linestyle = ':'
        elif 'psi' in algorithm_name:
            linestyle = '-.'
        elif 'rand' in algorithm_name:
            linestyle = '--'
        else:
            linestyle = '-'
        m1_t[algorithm_name].plot(ax=ax_bar, legend=None, secondary_y=False, linestyle=linestyle, color=ALL_ALG_COLORS[algorithm_name])

    ax_bar.yaxis.tick_left()
    ax_bar.yaxis.set_label_position("left")

    # x_labels = ['initial svd'] + list(range(1, len(r_periods)))
    x_labels = list(range(1, len(r_periods)))
    ax_bar.set_xticklabels(x_labels, rotation="horizontal")
    ax_bar.set_xlabel('period number')
    ax_bar.set_title('time spent for decomposition in seconds')
    ax_bar.spines[['right', 'top']].set_visible(False)

    handles_obj, labels_obj = ax_bar.get_legend_handles_labels()
    handles_metrics, labels_metrics = plt.gca().get_legend_handles_labels()
    handles_metrics_copy = [copy.copy(ha) for ha in handles_metrics]
    for i, label in enumerate(labels_metrics):
        labels_metrics[i] = label.replace(' (right)', '')
        # handles_metrics_copy[i].set_linewidth(2)

    fig.legend(handles_metrics_copy, labels_metrics, loc='lower center', ncol=len(handles_metrics), bbox_to_anchor=(0.55, 0.97))
    fig.add_artist(fig.legend(handles_obj, labels_obj, loc='center left', ncol=1, bbox_to_anchor=(-0.25, 0.5)))
    fig.tight_layout()
    # return fig

def visualize_relative_norm(metrics_dict, title=None, include_rsvd=True):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    rel_norms = {}
    for alg, metrics in metrics_dict.items():
        if not include_rsvd and alg == RAND_SVD:
            continue
        if alg not in [ALL_RANDOM, MOST_POPULAR]:
            rel_norms[alg] = metrics[-2]
    sns.lineplot(rel_norms, ax=ax, palette=ALL_ALG_COLORS)
    x_labels = [''] + ['initial svd'] + list(range(1, len(metrics[-2])))
    # x_labels = ['initial svd'] + list(range(1, len(metrics[-2])))
    ax.set_xticklabels(x_labels, rotation="horizontal")
    ax.set_xlabel('period number')
    ax.set_ylabel('relative error')
    # if title is not None:
    #     ax.set_title(title)
    # else:
    #     ax.set_title('$\\frac{\| A_t - U_t S_t V^T_t \|}{\| A_t \|}$')


def visualize_experiment_rank(alg_rank_metrics_dict, K=10, nostability=False):
    d = {metric: [] for metric in METRICS_NAMES_K(K)}
    d['Rank'] = []
    df_big = pd.DataFrame(d)
    for alg_name, rank_metrics_dict in alg_rank_metrics_dict.items():
        ranks = list(rank_metrics_dict.keys())
        dfm = pd.DataFrame(
            rank_metrics_dict[ranks[0]][0][:, 1:].T, 
            columns=METRICS_NAMES_K(K)).assign(Rank=ranks[0]).assign(Alg=alg_name)
        for rank in ranks[1:]:
            dfm = pd.concat([
                dfm, 
                pd.DataFrame(rank_metrics_dict[rank][0][:, 1:].T, columns=METRICS_NAMES_K(K)).assign(Rank=rank).assign(Alg=alg_name)
            ])
        df_big = pd.concat([df_big, dfm])
    mdf = pd.melt(df_big, id_vars=['Alg', 'Rank'], var_name='Metric') 
    mdf['Rank'] = mdf['Rank'].astype(int)
    gdf = mdf.groupby(['Alg', 'Rank', 'Metric'], as_index=False).sum()

    gdf['Metric'] = pd.Categorical(gdf['Metric'], METRICS_NAMES_K(K))
    gdf = gdf.sort_values("Metric")

    if nostability:
        gdf = gdf[gdf['Metric'] != METRICS_NAMES_K(K)[-1]]

    n_algs = len(alg_rank_metrics_dict.keys())
    # ylim = max(gdf['value']) * 1.05

    fig, ax = plt.subplots(1, N_METRICS, figsize=(6 * N_METRICS, 5))
    
    # for i, alg_name in enumerate(alg_rank_metrics_dict.keys()):
    #     sns.barplot(x="Rank", y="value", hue="Metric", 
    #                 data=gdf[gdf['Alg'] == alg_name],
    #                 ax=ax[i])
    #     ax[i].set_ylim(0, ylim)
    #     ax[i].set_title(alg_name)
    #     legend = ax[i].legend()
    #     legend.remove()

    for i, metric_name in enumerate(METRICS_NAMES_K(K)):
        sns.barplot(x="Rank", y="value", hue="Alg", 
                    data=gdf[gdf['Metric'] == metric_name],
                    ax=ax[i])
        # ax[i].set_ylim(0, ylim)
        ax[i].set_title(metric_name)
        # legend = ax[i].legend()
        # legend.remove()

    handles_obj, labels_obj = ax[0].get_legend_handles_labels()
    fig.legend(handles_obj, labels_obj, loc='upper center', ncol=len(handles_obj)) #, bbox_to_anchor=(0.5, 1.05))
    

def get_mdf(alg_rank_metrics_dict, K=10, validation=False):
    if validation:
        start_idx = 0
    else:
        start_idx = 1

    d = {metric: [] for metric in METRICS_NAMES_K(K)}
    d['Rank'] = []
    df_big = pd.DataFrame(d)
    for alg_name, rank_metrics_dict in alg_rank_metrics_dict.items():
        if alg_name in [ALL_RANDOM, MOST_POPULAR]:
            continue
        if type(rank_metrics_dict) == type({}):
            ranks = list(rank_metrics_dict.keys())
        else:
            rank_metrics_dict = [rank_metrics_dict]
            ranks = [0]
        dfm = pd.DataFrame(
            rank_metrics_dict[ranks[0]][0][:, start_idx:].T, 
            columns=METRICS_NAMES_K(K)).assign(Rank=ranks[0]).assign(Alg=alg_name)
        for rank in ranks[1:]:
            dfm = pd.concat([
                dfm, 
                pd.DataFrame(rank_metrics_dict[rank][0][:, start_idx:].T, columns=METRICS_NAMES_K(K)).assign(Rank=rank).assign(Alg=alg_name)
            ])
        df_big = pd.concat([df_big, dfm])
    # mdf = pd.melt(df_big, id_vars=['Alg', 'Rank'], var_name='Metric') 
    # mdf['Rank'] = mdf['Rank'].astype(int)
    return df_big
    # gdf = mdf.groupby(['Alg', 'Rank', 'Metric'], as_index=False).sum()
    # return gdf


def visualize_for_pdf(bdf, K=10):
    sum_df = bdf.loc[:, ~bdf.columns.isin(['period', 'init_alg'])].groupby(['Alg', 'id_run', 'Rank']).sum().reset_index()
    sum_metrics_df = sum_df.loc[:, ~sum_df.columns.isin(['id_run', 'releps', 'seconds'])]
    gdf = pd.melt(sum_metrics_df, id_vars=['Alg', 'Rank'], var_name='Metric') 
    allalgnames = [ x for x in ALL_ALG_NAMES[:-2] if not (("ort" in x) and ("reused" in x))] 
    gdf = gdf[gdf['Alg'].isin(allalgnames)]

    gdf['Alg'] = pd.Categorical(gdf['Alg'], allalgnames)
    gdf = gdf.sort_values("Alg")
    
    fig, ax = plt.subplots(1, 2, figsize=(6 * 2, 5))
    for i, metric_name in enumerate(['NDCG@10', 'stability@10']):
        axi = ax[i]
        sns.barplot(x="Rank", y="value", hue="Alg", 
                    data=gdf[gdf['Metric'] == metric_name],
                    ax=axi,
                    palette=ALL_ALG_COLORS)
        axi.set_ylabel('')
        axi.set_title(metric_name, fontsize=14)
        legend = axi.legend()
        legend.remove()
        for label in (axi.get_xticklabels() + axi.get_yticklabels()):
            label.set_fontsize(12)
        axi.axes.get_xaxis().set_visible(False)

    # ax[-1].set_visible(False)
    handles_obj, labels_obj = ax[0].get_legend_handles_labels()
    fig.legend(handles_obj, labels_obj, ncol=1, bbox_to_anchor=(1.25, 1.0), prop={'size': 14})
    plt.tight_layout()
    return gdf

def visualize_small_cummulative(bdf, K=10):

    sum_df = bdf.loc[:, ~bdf.columns.isin(['period', 'init_alg'])].groupby(['Alg', 'id_run', 'Rank']).sum().reset_index()
    sum_metrics_df = sum_df.loc[:, ~sum_df.columns.isin(['id_run', 'releps', 'seconds'])]
    gdf = pd.melt(sum_metrics_df, id_vars=['Alg', 'Rank'], var_name='Metric') 
    allalgnames = [ x for x in ALL_ALG_NAMES[:-2] if not (("ort" in x) and ("reused" in x))] 
    gdf = gdf[gdf['Alg'].isin(allalgnames)]

    gdf['Alg'] = pd.Categorical(gdf['Alg'], allalgnames)
    gdf = gdf.sort_values("Alg")
    
    fig, ax = plt.subplots(1, 2, figsize=(6 * 2, 5))
    for i, metric_name in enumerate(['NDCG@10', 'stability@10']):
        axi = ax[i]
        sns.barplot(x="Rank", y="value", hue="Alg", 
                    data=gdf[gdf['Metric'] == metric_name],
                    ax=axi,
                    palette=ALL_ALG_COLORS)
        axi.set_ylabel('')
        axi.set_title(metric_name, fontsize=14)
        legend = axi.legend()
        legend.remove()
        for label in (axi.get_xticklabels() + axi.get_yticklabels()):
            label.set_fontsize(12)
        axi.axes.get_xaxis().set_visible(False)

    # ax[-1].set_visible(False)
    handles_obj, labels_obj = ax[0].get_legend_handles_labels()
    fig.legend(handles_obj, labels_obj, ncol=1, bbox_to_anchor=(1.25, 1.0), prop={'size': 14})
    plt.tight_layout()
    return gdf


def visualize_cummulative(bdf, K=10, validation=False):

    if validation:
        gdf = pd.melt(bdf.loc[:, ~bdf.columns.isin(['id_run', 'seconds', 'releps', 'init_alg'])], id_vars=['Alg', 'Rank'], var_name='Metric')
        allalgnames = [SVD, RAND_SVD_P1, RAND_SVD_P2, RAND_SVD_P3]
        # allalgnames = [SVD, RAND_SVD, RAND_SVD_P1, RAND_SVD_P2]
    else:
        sum_df = bdf.loc[:, ~bdf.columns.isin(['period', 'init_alg'])].groupby(['Alg', 'id_run', 'Rank']).sum().reset_index()
        sum_metrics_df = sum_df.loc[:, ~sum_df.columns.isin(['id_run', 'releps', 'seconds'])]
        gdf = pd.melt(sum_metrics_df, id_vars=['Alg', 'Rank'], var_name='Metric') 
        allalgnames = [ x for x in ALL_ALG_NAMES[:-2] if not (("ort" in x) and ("reused" in x))] 
        gdf = gdf[gdf['Alg'].isin(allalgnames)]

    gdf['Alg'] = pd.Categorical(gdf['Alg'], allalgnames)
    gdf = gdf.sort_values("Alg")

    if validation:
        fig, ax = plt.subplots(2, 2, figsize=(5 * (N_METRICS - 1), 13))
        for i, metric_name in enumerate(METRICS_NAMES_K(K)[:-1]):
            if i <= 1:
                axi = ax[0][i]
            else:
                axi = ax[1][i % 2]
            sns.barplot(x="Rank", y="value", hue="Alg", 
                        data=gdf[gdf['Metric'] == metric_name],
                        ax=axi,
                        palette=ALL_ALG_COLORS)
            axi.set_title(metric_name)
            legend = axi.legend()
            legend.remove()
            axi.set_ylabel('')
            axi.set_xlabel('rank', fontsize=12)
            axi.set_title(metric_name, fontsize=14)
            for label in (axi.get_xticklabels() + axi.get_yticklabels()):
                label.set_fontsize(12)
        handles_obj, labels_obj = axi.get_legend_handles_labels()
        fig.legend(handles_obj, labels_obj, loc='upper center', ncol=len(handles_obj), prop={'size': 16}, bbox_to_anchor=(0.5, 1.02))
        plt.tight_layout()
        return
    
    fig, ax = plt.subplots(2, 3, figsize=(6 * 3, 5 * 2))
    for i, metric_name in enumerate(METRICS_NAMES_K(K)):
        if i <= 2:
            axi = ax[0][i]
        else:
            axi = ax[1][i % 3]
        sns.barplot(x="Rank", y="value", hue="Alg", 
                    data=gdf[gdf['Metric'] == metric_name],
                    ax=axi,
                    palette=ALL_ALG_COLORS)
        axi.set_ylabel('')
        axi.set_title(metric_name, fontsize=14)
        legend = axi.legend()
        legend.remove()
        for label in (axi.get_xticklabels() + axi.get_yticklabels()):
            label.set_fontsize(12)
        axi.axes.get_xaxis().set_visible(False)

    ax[-1][-1].set_visible(False)
    handles_obj, labels_obj = ax[0][0].get_legend_handles_labels()
    fig.legend(handles_obj, labels_obj, ncol=1, bbox_to_anchor=(0.94, 0.5), prop={'size': 16})
    plt.tight_layout()
    return gdf


def visualize_boxplots(alg_rank_metrics_dict, K=10, validation=True):
    if validation:
        start_idx = 0
    else:
        start_idx = 1

    d = {metric: [] for metric in METRICS_NAMES_K(K)}
    d['Rank'] = []
    df_big = pd.DataFrame(d)
    for alg_name, rank_metrics_dict in alg_rank_metrics_dict.items():
        ranks = list(rank_metrics_dict.keys())
        dfm = pd.DataFrame(
            rank_metrics_dict[ranks[0]][0][:, start_idx:].T, 
            columns=METRICS_NAMES_K(K)).assign(Rank=ranks[0]).assign(Alg=alg_name)
        for rank in ranks[1:]:
            dfm = pd.concat([
                dfm, 
                pd.DataFrame(rank_metrics_dict[rank][0][:, start_idx:].T, columns=METRICS_NAMES_K(K)).assign(Rank=rank).assign(Alg=alg_name)
            ])
        df_big = pd.concat([df_big, dfm])
    mdf = pd.melt(df_big, id_vars=['Alg', 'Rank'], var_name='Metric') 
    mdf['Rank'] = mdf['Rank'].astype(int)
    fig, ax = plt.subplots(1, N_METRICS, figsize=(6 * N_METRICS, 5))
    for i, metric_name in enumerate(METRICS_NAMES_K(K)):
        sns.boxplot(x="Rank", y="value", hue="Alg", 
                    data=mdf[mdf['Metric'] == metric_name],
                    ax=ax[i],
                    palette=ALL_ALG_COLORS)
        legend = ax[i].legend()
        legend.remove()
        ax[i].set_title(metric_name)
    
    handles_obj, labels_obj = ax[0].get_legend_handles_labels()
    fig.legend(handles_obj, labels_obj, loc='upper center', ncol=len(handles_obj)) #, bbox_to_anchor=(0.5, 1.05))
    


def visualize_split_metrics(algorithm_name, metrics, K=10):
    fig, ax = plt.subplots(1, N_METRICS, figsize=(6 * N_METRICS, 5))
    old_color, new_color = ['salmon', 'olivedrab']
    for metric_name_idx, metric_name in enumerate(METRICS_NAMES):

        total_metrics = metrics[0][metric_name_idx]
        initial_metrics = metrics[1][metric_name_idx]
        new_metrics = metrics[2][metric_name_idx]

        if metric_name_idx != len(METRICS_NAMES) - 1:
            ax[metric_name_idx].stackplot(
                range(len(new_metrics)), 
                initial_metrics, new_metrics, 
                labels=['initial_metrics','new_metrics'],
                colors=[old_color, new_color], alpha=0.5)
            ax[metric_name_idx].plot(new_metrics, linewidth=3, color = new_color)
        else:
            all_total_unique_recs = [list(x[1]) for x in metrics[-1][METRICS_TYPES[0]]]
            all_initial_unique_recs = [list(x[1]) for x in metrics[-1][METRICS_TYPES[1]]]
            all_new_unique_recs = [list(x[1]) for x in metrics[-1][METRICS_TYPES[2]]]

            num_intersection = []
            num_new = []
            num_old = []
            for period in range(len(metrics[-1][METRICS_TYPES[0]])):
                total_unique_recs = set(all_total_unique_recs[period])
                initial_unique_recs = set(all_initial_unique_recs[period])
                new_unique_recs = set(all_new_unique_recs[period])

                num_intersection.append(len(initial_unique_recs & new_unique_recs))
                num_new.append(len(new_unique_recs - initial_unique_recs))
                num_old.append(len(initial_unique_recs - new_unique_recs))
            
            ax[metric_name_idx].stackplot(
                range(len(new_metrics)), 
                num_intersection, num_old, num_new,
                labels=['intersection', 'only_old', 'only_new'],
                colors=['darkgrey', old_color, new_color],
                alpha=0.5)
            
            secax = ax[metric_name_idx].twinx()
            secax.plot(total_metrics, linewidth=3, label='total_metrics')
            secax.plot(initial_metrics, linewidth=3, color=old_color, label='initial_metrics')
            secax.plot(new_metrics, linewidth=3, color=new_color, label='new_metrics')
            plt.grid(False)
            handles_c, labels_c = ax[-1].get_legend_handles_labels()
            secax.legend(loc='lower right')
        ax[metric_name_idx].legend(loc='upper left')
        ax[metric_name_idx].set_title(METRICS_NAMES[metric_name_idx](K))

    fig.suptitle(f"{algorithm_name}")


def visualize_all(algorithm_metrics_dict, K, split_data):
    (
        r_periods,
        r_total_users_items,
        r_users_ids,
        r_A_act,
        r_users_actual_watched,
        r_clean_next_portion,
        r_dirt,
        r_n_virtual_users,
        r_n_real_active,
        r_n_pure_cold,
        training,
        holdout
    ) = split_data

    algorithm_names = list(algorithm_metrics_dict.keys())
    algorithm_metrics = [metrics[0] for metrics in algorithm_metrics_dict.values()]

    # delta_users = []
    # delta_items = []
    # for i in range(1, len(r_total_users_movies_start)):
    #     delta_users.append(r_total_users_movies_start[i][0] - r_total_users_movies_start[i - 1][0])
    #     delta_items.append(r_total_users_movies_start[i][1] - r_total_users_movies_start[i - 1][1])

    # all_users = [x[0] for x in r_total_users_movies_start[:-1]]
    # all_movies = [x[1] for x in r_total_users_movies_start[:-1]]
    width = .2 # width of a bar

    fig, ax_bar = plt.subplots(1, N_METRICS, figsize=(6 * N_METRICS, 5))

    real_active = 'number of real active users'
    virtual = 'number of created virtual users'
    for metric_idx in range(N_METRICS):
        info_dict = {
            # 'total_users' : [np.unique(users_ids) for users_ids in r_users_ids],
            # 'total_movies' : [np.unique(users_ids) for users_ids in r_users_ids],
            real_active: r_n_real_active,
            virtual: r_n_virtual_users
        }
        for alg_idx, algorithm_name in enumerate(algorithm_names):
            info_dict[algorithm_name] = algorithm_metrics[alg_idx][metric_idx]

        m1_t = pd.DataFrame(info_dict)

        user_movie_colors = ['tab:olive', 'tab:pink']
        real_virtual = ['tab:green', 'tab:gray']

        m1_t[[real_active, virtual]].plot(ax=ax_bar[metric_idx], kind='bar', color=real_virtual, secondary_y=True, stacked=True, legend=None, alpha=0.3)
        for container in ax_bar[metric_idx].containers[1:]:
            ax_bar[metric_idx].bar_label(container, fontsize=7, label_type='center')

        # m1_t[['total_users','total_movies']].plot(ax=ax_bar[metric_idx], kind='bar', color=user_movie_colors, secondary_y=True, legend=None, alpha=0.5, width=width)

        nbars = m1_t[[real_active, virtual]].shape[1]
        for container in ax_bar[metric_idx].containers[nbars:]:
            ax_bar[metric_idx].bar_label(container, fontsize=7, label_type='edge')

        # plt.ylabel('# objects used for decomosition')
        # plt.xlabel('period number')

        for alg_idx, algorithm_name in enumerate(algorithm_names):
            if algorithm_name == ALL_RANDOM:
                linestyle = '-.'
            elif algorithm_name == MOST_POPULAR:
                linestyle = '-.'
            elif algorithm_name == SVD:
                linestyle = '--'
            else:
                linestyle = '-'
            m1_t[algorithm_name].plot(ax=ax_bar[metric_idx], legend=None, secondary_y=False, linestyle=linestyle, color=ALL_ALG_COLORS[algorithm_name])

        ax_bar[metric_idx].yaxis.tick_left()
        ax_bar[metric_idx].yaxis.set_label_position("left")

        x_labels = ['initial svd'] + list(range(1, len(r_periods)))
        ax_bar[metric_idx].set_xticklabels(x_labels, rotation="horizontal")
        ax_bar[metric_idx].set_xlabel('period number')
        ax_bar[metric_idx].set_title(METRICS_NAMES[metric_idx](K))
        ax_bar[metric_idx].spines[['right', 'top']].set_visible(False)
        

    handles_obj, labels_obj = ax_bar[0].get_legend_handles_labels()
    handles_metrics, labels_metrics = plt.gca().get_legend_handles_labels()
    handles_metrics_copy = [copy.copy(ha) for ha in handles_metrics]
    for i, label in enumerate(labels_metrics):
        labels_metrics[i] = label.replace(' (right)', '')
        # handles_metrics_copy[i].set_linewidth(2)
    
    fig.legend(handles_metrics_copy, labels_metrics, loc='lower center', ncol=len(handles_metrics), bbox_to_anchor=(0.5, 0.97))
    fig.add_artist(fig.legend(handles_obj, labels_obj, loc='upper center', ncol=len(handles_obj), bbox_to_anchor=(0.5, 0.03)))
    fig.tight_layout()
    # return fig
