import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn import metrics


def exp_decay_factor(k,rate=0.5):
    return (1-rate)**k

def compute_score(metric_vals):
    score = 0
    for i,val in enumerate(metric_vals):
        score += val*exp_decay_factor(i)

    return score

def load_metrics(parent_folder='./',k=5):
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color',plt.cm.Accent(np.linspace(0,1,5)))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    y_min = np.infty
    y_max = 0
    fig, axs = plt.subplots(1, 4)
    folders = sorted([i for i in glob.glob(os.path.join(parent_folder, 'results*')) if os.path.isdir(i)])

    score_dict = {
        'method':'',
        'accuracy':[],
        'precision':[],
        'recall':[],
        'f_score':[],
    }

    for i,folder in enumerate(folders):
        prf_metrics = np.load(os.path.join(folder, 'prf_metrics.npy'),allow_pickle=True).item()
        axs[0].plot(np.arange(1,k+1),prf_metrics['precisions'][:k],color = colors[i],label=folder)
        axs[1].plot(np.arange(1,k+1),prf_metrics['recalls'][:k],color = colors[i],label=folder)
        axs[2].plot(np.arange(1,k+1),prf_metrics['f_scores'][:k],color = colors[i],label=folder)
        axs[3].plot(np.arange(1,k+1),prf_metrics['accs'][:k],color = colors[i],label=folder)

        curr_y_min = np.min(np.concatenate((prf_metrics['precisions'],prf_metrics['recalls'],
                                            prf_metrics['f_scores'],prf_metrics['accs'])))
        curr_y_max = np.max(np.concatenate((prf_metrics['precisions'],prf_metrics['recalls'],
                                            prf_metrics['f_scores'],prf_metrics['accs'])))
        if curr_y_min < y_min:
            y_min = curr_y_min
        if curr_y_max > y_max:
            y_max = curr_y_max

        print('-----')

        # compute score
        score_dict['method']=folder
        score_dict['accuracy']=compute_score(prf_metrics['accs'][:k])
        score_dict['precision']=compute_score(prf_metrics['precisions'][:k])
        score_dict['recall']=compute_score(prf_metrics['recalls'][:k])
        score_dict['f_score']=compute_score(prf_metrics['f_scores'][:k])

        cluster_metrics = compute_cluster_metrics(prf_metrics)
        score_dict['cluster_metrics']=cluster_metrics

        print(score_dict)

    for ax in axs:
        ax.set_xlim(1,k)
        # ax.set_ylim(0.99*y_min,1.01*y_max)
        leg = ax.legend(loc='right',fontsize='x-small')
        leg.set_draggable(state=True)
    plt.show()

def compute_cluster_metrics(metrics_dict):
    labels_pred = metrics_dict['top_k_pred_labels'][:,0]
    labels_gt = metrics_dict['top_k_gt_labels'][:,0]

    cluster_metrics = {}
    cluster_metrics['homogeneity'] = metrics.homogeneity_score(labels_gt,labels_pred)
    cluster_metrics['completeness'] = metrics.completeness_score(labels_gt,labels_pred)
    cluster_metrics['v_measure'] = metrics.v_measure_score(labels_gt,labels_pred)
    cluster_metrics['adjusted_rand_score'] = metrics.adjusted_rand_score(labels_gt,labels_pred)
    cluster_metrics['adjusted_mutual_info_score'] = metrics.adjusted_mutual_info_score(labels_gt,labels_pred)

    return cluster_metrics

if __name__ == "__main__":
    load_metrics()