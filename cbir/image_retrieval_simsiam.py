import argparse
import numpy as np
import math
import scipy.spatial as spatial
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, \
    precision_recall_fscore_support, accuracy_score
from sklearn.decomposition import PCA

from model import *
from sim_siam_data_loader import *


ID_DIM = [2,2,2,4]
NUM_CLASSES = np.prod(ID_DIM)

def distance(point1, database):
    dst = []
    for i in range(database.shape[0]):
        dst.append(np.linalg.norm(point1-database[i,:]))
    return dst

def hamming_distance(point1, database):
    dst = []
    for i in range(database.shape[0]):
        dst.append(spatial.distance.hamming(point1, database[i, :]))
    return dst

def hashing(x,bits,w,b,t):
    m = len(x)
    a = np.inner(w,x)+b
    f = 0.5*(1+np.sign(np.cos(a)+t))
    return f

def dice_score(set1, set2):
    a = set1.intersection(set2)
    return (2*len(a))/(len(set1)+len(set2))

def jaccard_score(set1, set2):
    b = set1.intersection(set2)
    c = set.union(set2)
    return (len(b))/(len(c))

def comb(n, k):
    return math.factorial(n)/math.factorial(k)/math.factorial(n-k)

def decimal_to_binary(n):
    bin_str = '{0:03b}'.format(int(n))
    bin_list = [int(x) for x in bin_str]
    return bin_list

def compute_img_flips(img,ndims = 3): # b, c, x, y, z
    ncombs = 2**ndims

    list_images = []

    for i in range(ncombs):
        flip_axes_mask = decimal_to_binary(i)
        flipped_img = copy.deepcopy(img).cpu().detach().numpy()
        for a in range(len(flip_axes_mask)):
            if flip_axes_mask[a] == True:
                flipped_img = np.flip(flipped_img, axis=a+2)

        list_images.append(torch.from_numpy(copy.deepcopy(flipped_img)).float())

    return list_images

def load_query_info(model, query_data_loader, database_data_loader, device, bits = 10000, verbose = False):
    query_info = {'input1s':[],
                  'input2s': [],
                  'input3s': [],
                  'recons': [],
                  'zs':[],
                  'post_zs':[],
                  'distances':[],
                  'hamming_distances':[],
                  'filepaths':[],
                  'hashed_embeddings':[],
                  'labels':[]
                  }
    database_info = {'input1s':[],
                     'input2s': [],
                     'input3s': [],
                     'recons': [],
                     'zs':[],
                     'post_zs':[],
                     'distances':[],
                     'hamming_distances':[],
                     'filepaths':[],
                     'hashed_embeddings':[],
                     'labels':[]
                     }
    model.eval()
    with torch.no_grad():
        for iter,batch in enumerate(query_data_loader):
            input1, input2, input3, pos_img, neg_img = batch[0].to(device).float(), batch[1].to(device).float(), \
                                               batch[2].to(device).float(), \
                                               batch[3].to(device).float(), batch[4].to(device).float()
            filepath = batch[-1][0]

            p1, p2, z1, z2, input1, recon, z_pos, z_neg = model(input1, input3, pos_img, neg_img)
            input1 = input1.cpu().detach().numpy()
            input2 = input2.cpu().detach().numpy()
            input3 = input3.cpu().detach().numpy()
            recon = recon.cpu().detach().numpy()
            z1 = z1.cpu().detach().numpy()
            post_z1 = post_process_latents(z1)

            query_info['input1s'].append(input1)
            query_info['input2s'].append(input2)
            query_info['input3s'].append(input3)
            query_info['recons'].append(recon)
            query_info['zs'].append(z1)
            query_info['post_zs'].append(post_z1)
            query_info['filepaths'].append(filepath)

            # hashing params
            encoded_query = z1
            encoded_query = np.squeeze(encoded_query)
            m = len(encoded_query)
            b = np.random.uniform(-math.pi, math.pi, bits)
            w = 0.001 * np.random.randn(bits, m)
            t = np.random.uniform(-1, 1, bits)

            query_info['hashed_embeddings'].append(hashing(encoded_query, bits, w, b, t))

            if verbose:
                print("===> Query {}/{}".format(iter+1,len(query_data_loader)))
                print("Filepath: {}".format(filepath))

        for iter,batch in enumerate(database_data_loader):
            input1, input2, input3, pos_img, neg_img = batch[0].to(device).float(), batch[1].to(device).float(), \
                                               batch[2].to(device).float(), \
                                               batch[3].to(device).float(), batch[4].to(device).float()
            filepath = batch[-1][0]

            p1, p2, z1, z2, input1, recon, z_pos, z_neg = model(input1, input3, pos_img, neg_img)
            input1 = input1.cpu().detach().numpy()
            input2 = input2.cpu().detach().numpy()
            input3 = input3.cpu().detach().numpy()
            recon = recon.cpu().detach().numpy()
            z1 = z1.cpu().detach().numpy()
            post_z1 = post_process_latents(z1)

            database_info['input1s'].append(input1)
            database_info['input2s'].append(input2)
            database_info['input3s'].append(input3)
            database_info['recons'].append(recon)
            database_info['zs'].append(z1)
            database_info['post_zs'].append(post_z1)
            database_info['filepaths'].append(filepath)

            # hashing params
            encoded_database = z1
            encoded_database = np.squeeze(encoded_database)
            m = len(encoded_database)
            b = np.random.uniform(-math.pi, math.pi, bits)
            w = 0.001 * np.random.randn(bits, m)
            t = np.random.uniform(-1, 1, bits)

            database_info['hashed_embeddings'].append(hashing(encoded_database, bits, w, b, t))

            if verbose:
                print("===> Database {}/{}".format(iter+1,len(database_data_loader)))
                print("Filepath: {}".format(filepath))

    query_info['labels'] = get_labels(query_info)
    database_info['labels'] = get_labels(database_info)

    return query_info, database_info

def get_image_retrieval_loader(query_bs = 1, database_bs = 1, num_works=0):
    query_dataset = SimSiamDataset(num_samples_per_epoch=None, phase='query')
    database_dataset = SimSiamDataset(num_samples_per_epoch=None, phase='database')

    query_data_loader = data.DataLoader(dataset=query_dataset, batch_size=query_bs, shuffle=False,
                                           num_workers=num_works,
                                           pin_memory=False)
    database_data_loader = data.DataLoader(dataset=database_dataset, batch_size=database_bs, shuffle=False,
                                           num_workers=num_works,
                                           pin_memory=False)

    return query_data_loader, database_data_loader

def plot_latent_space(query_info,database_info,output_folder):
    tsne = TSNE(n_components=2)
    database_latents = np.squeeze(np.array(database_info['post_zs']))
    query_latents = np.squeeze(np.array(query_info['post_zs']))

    database_tsne = tsne.fit_transform(database_latents)
    query_tsne = tsne.fit_transform(query_latents)

    database_category = []
    query_category = []
    colors = {'head_and_neck':'red',
              'prostate':'green'}

    for i in range(len(database_info['filepaths'])):
        fp = database_info['filepaths'][i]
        fp = fp.split('/')
        if fp[-1].startswith('HN') or fp[-1].startswith('pt'): # if head and neck
            database_category.append('head_and_neck')
        else:
            database_category.append('prostate')


    for i in range(len(query_info['filepaths'])):
        fp = query_info['filepaths'][i]
        fp = fp.split('/')
        if fp[-1].startswith('HN') or fp[-1].startswith('pt'): # if head and neck
            query_category.append('head_and_neck')
        else:
            query_category.append('prostate')

    database_df = pd.DataFrame(dict(z0=database_tsne[:,0], z1=database_tsne[:,1], category=database_category))
    query_df = pd.DataFrame(dict(z0=query_tsne[:,0], z1=query_tsne[:,1], category=query_category))

    fig1, ax1 = plt.subplots()
    ax1.scatter(database_df['z0'],database_df['z1'],
               c=database_df['category'].map(colors), marker='o')
    ax1.scatter(query_df['z0'],query_df['z1'],
               c=query_df['category'].map(colors), marker='^')


    fig2, ax2 = plt.subplots()
    ax2.scatter(query_df['z0'],query_df['z1'],
               c=query_df['category'].map(colors), marker='^')

    fig1.savefig(os.path.join(output_folder, 'all_latents.png'),
                bbox_inches='tight')
    fig2.savefig(os.path.join(output_folder, 'query_latents.png'),
                bbox_inches='tight')



def get_middle_factors(x):
    factors = []
    for i in range(1,x+1):
        if x%i == 0:
            factors.append(i)

    id = len(factors)//2-1
    factor1 = factors[id]
    factor2 = x/factor1
    return [int(factor1),int(factor2)]


def plot_query_and_retrieved_img(query_imgs,retrieved_imgs,query_files,retrieved_files,output_folder,num_plots = 50):

    num_imgs = len(query_imgs)
    num_slices = query_imgs[0].shape[-1]
    slice_spacing = np.round(np.linspace(0,num_slices-1,num_plots))
    seg_cmap = mpl.cm.gnuplot
    bounds = [0.5,1.5,2.5,3.5]
    norm = mpl.colors.BoundaryNorm(bounds,seg_cmap.N)
    r,c = get_middle_factors(num_plots)

    for i in range(num_imgs):
        fig, axs = plt.subplots(2*r,c)
        for j in range(len(slice_spacing)):
            m,n = np.unravel_index(j,(r,c))
            axs[m,n].imshow(query_imgs[i][0,0,:,:,int(slice_spacing[j])],
                            cmap='gray',vmin=0,vmax=1)
            axs[m,n].imshow(query_imgs[i][0,1,:,:,int(slice_spacing[j])],
                            cmap=seg_cmap,norm=norm,vmin=0,vmax=3,alpha=0.5)
            axs[r+m,n].imshow(retrieved_imgs[i][0,0,:,:,int(slice_spacing[j])],
                            cmap='gray',vmin=0,vmax=1)
            axs[r+m,n].imshow(retrieved_imgs[i][0,1,:,:,int(slice_spacing[j])],
                            cmap=seg_cmap,norm=norm,vmin=0,vmax=3,alpha=0.5)

            axs[m,n].set_xticks([])
            axs[m,n].set_yticks([])
            axs[r+m,n].set_xticks([])
            axs[r+m,n].set_yticks([])

        query_file = query_files[i].split('/')[-1].split('.nii.gz')[0]
        retrieved_file = retrieved_files[i].split('/')[-1].split('.nii.gz')[0]
        fig.savefig(os.path.join(output_folder,'query_'+query_file+'_retrieved_'+retrieved_file+'.png'),
                    dpi=500,bbox_inches='tight')



def get_labels(info_dict):
    labels = []
    for i in range(len(info_dict['filepaths'])):
        fp = info_dict['filepaths'][i]
        category = get_category_of_file(fp)
        labels.append(category)
    return labels

def get_category_of_file(filepath, id_dim = ID_DIM, ext = '.nii.gz'):
    fp = filepath.split(ext)[0]
    label_file = fp.split('/')[:-1]+['labels']+[fp.split('/')[-1]]
    label_file = '/'.join(label_file) + '.txt'
    assert os.path.exists(label_file), 'Label file "'+label_file+'" does not exist'
    with open(label_file) as f:
        lines = f.readlines()
    label_id = lines[0].replace('\n', '')
    label_id = label_id.replace('\t', '')
    category = np.ravel_multi_index(tuple([int(x) for x in list(label_id)]), dims = id_dim, order='C')
    # to get label id back from category, used "np.unravel_index(np.prod(id_dim)-1,shape=id_dim,order='C')"

    return category

def compute_cluster_score(info_dict,output_folder):

    latents = np.squeeze(np.array(info_dict['post_zs']))
    labels = np.array(info_dict['labels'])
    s_score = silhouette_score(latents,labels)
    ch_score = calinski_harabasz_score(latents,labels)
    db_score = davies_bouldin_score(latents,labels)

    output_metrics = {
        'silhouette': s_score,
        'calinski_harabasz': ch_score,
        'davies_bouldin': db_score
    }

    print('Latents shape:',latents.shape)
    print('Labels shape:',labels.shape)
    print('Silhouette score:',s_score)
    print('Calinski-Harabasz score:',ch_score)
    print('Davies-Bouldin score:',db_score)
    np.save(os.path.join(output_folder,'cluster_metrics.npy'),output_metrics)

    return output_metrics

def post_process_latents(latents):
    '''
    Implement post processing on latents
    :param latents: (n_samples, n_features)
    :return:
    '''
    z_norm = latents / np.linalg.norm(latents, axis = -1, keepdims=True)
    # pca = PCA(whiten=True)
    # z_wn = pca.fit_transform(z_norm)
    # z_nwn = z_wn / np.linalg.norm(z_wn, axis = -1, keepdims=True)
    return z_norm


def get_topk(query_label, database_info, sorted_dists, top_k):
    top_k_ids = sorted_dists[:, 0][:top_k]
    pred_top_k_labels = [database_info['labels'][int(id)] for id in top_k_ids]
    gt_top_k_labels = [query_label]*top_k
    return pred_top_k_labels, gt_top_k_labels

def compute_prf_metrics(top_k_pred_labels,top_k_gt_labels,output_folder):
    '''
    plot precision, recall, and f-score as a function of k
    :param top_k_pred_labels: list of pred labels (length k)
    :param top_k_gt_labels: list of ground truth labels (length k)
    :param output_folder: results folder
    :return:
    '''
    top_k_pred_labels = np.array(top_k_pred_labels) # (n_samples, k)
    top_k_gt_labels = np.array(top_k_gt_labels) # (n_samples, k)
    precisions = []
    recalls = []
    f_scores = []
    supports = []
    accs = []
    for i in range(top_k_pred_labels.shape[1]):
        y_pred = np.ravel(top_k_pred_labels[:,:i+1], order='C')
        y_true = np.ravel(top_k_gt_labels[:,:i+1], order='C')
        precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred,
                                                                              labels=list(np.arange(NUM_CLASSES)),
                                                                              zero_division=1)
        acc = accuracy_score(y_true,y_pred)
        precisions.append(np.mean(precision[support>0]))
        recalls.append(np.mean(recall[support>0]))
        f_scores.append(np.mean(f_score[support>0]))
        supports.append(support)
        accs.append(acc)

    output_metrics = {
        'precisions': precisions,
        'recalls': recalls,
        'f_scores': f_scores,
        'supports': supports,
        'accs': accs,
        'top_k_pred_labels': top_k_pred_labels,
        'top_k_gt_labels': top_k_gt_labels,
    }

    print('-----Top 1 to {}-----'.format(top_k_pred_labels.shape[1]))
    print('Precision:', precisions)
    print('Recall:', recalls)
    print('F Score:', f_scores)
    print('Accuracy:', accs)
    print('Support:', supports)
    np.save(os.path.join(output_folder,'prf_metrics.npy'),output_metrics)

    return output_metrics

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Content Based Image Retrieval')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--model_pth', type=str, default='./model_checkpoints_simsiam/final_model.pth',
                        help='Model path for loading weights. Default=./model_checkpoints_simsiam/final_model.pth')
    parser.add_argument('--top_k', type=int, default=30,
                        help='Top k related images are retrieved. ')
    parser.add_argument('--output_folder', type=str, default='./results_simsiam',
                        help='Output folder for saving figures and metrics. Default=./results_simsiam')
    opt = parser.parse_args()

    print(opt)


    torch.manual_seed(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('===> Loading datasets')
    query_data_loader, database_data_loader = get_image_retrieval_loader()

    print('===> Building model')
    model = SimSiam(in_channels=2, latent_dim=1024).to(device)

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)
    if os.path.isfile(opt.model_pth):
        model = torch.load(opt.model_pth)
        print('===> Loading saved model')

    top_k = opt.top_k
    bits_levels = [10000]
    dice_per_bits_level = []
    jaccard_per_bits_level = []
    query_files = []
    retrieved_files = []
    retrieved_files_hash = []
    query_imgs = []
    retrieved_imgs = []
    retrieved_imgs_hash = []
    pred_labels = []
    gt_labels = []
    for bits in bits_levels:
        query_info, database_info = load_query_info(model, query_data_loader, database_data_loader, device, bits=bits)
        ds = []
        js = []
        for q in range(len(query_info['filepaths'])):
            qhe = query_info['hashed_embeddings'][q]
            dhe = np.array(database_info['hashed_embeddings'])

            qe = query_info['post_zs'][q]
            de = np.array(database_info['post_zs'])

            # compute distances between query and database
            euc_distances = distance(qe,de)
            ham_distances = hamming_distance(qhe,dhe)

            # sort by lowest distance
            euc_distances = list(enumerate(euc_distances))
            euc_distances = np.array(sorted(euc_distances, key=lambda x: x[1])) # order in first col and value in second
            ham_distances = list(enumerate(ham_distances))
            ham_distances = np.array(sorted(ham_distances, key=lambda x: x[1])) # order in first col and value in second

            # compute dice and jaccard
            ds.append(dice_score(set(ham_distances[:,0][:top_k]),set(euc_distances[:,0][:top_k])))
            js.append(jaccard_score(set(ham_distances[:,0][:top_k]),set(euc_distances[:,0][:top_k])))

            # get filepaths
            query_file = query_info['filepaths'][q]
            closest_file_idx = int(euc_distances[:,0][0])
            closest_file_idx_hash = int(ham_distances[:,0][0])

            retrieved_file = database_info['filepaths'][closest_file_idx]
            retrieved_file_hash = database_info['filepaths'][closest_file_idx_hash]

            query_files.append(query_file)
            retrieved_files.append(retrieved_file)
            retrieved_files_hash.append(retrieved_file_hash)

            query_imgs.append(query_info['input1s'][q])
            retrieved_imgs.append(database_info['input1s'][closest_file_idx])
            retrieved_imgs_hash.append(database_info['input1s'][closest_file_idx_hash])

            print('Query path: {}'.format(query_file))
            print('Retrieved path (euclidean): {}'.format(retrieved_file))
            # print('Retrieved path (hashing): {}'.format(retrieved_file_hash))

            pred_labels_top_k, gt_labels_top_k = get_topk(query_info['labels'][q], database_info, euc_distances, top_k)
            pred_labels.append(pred_labels_top_k)
            gt_labels.append(gt_labels_top_k)

        dice_per_bits_level.append(ds)
        jaccard_per_bits_level.append(js)

    compute_cluster_score(query_info, output_folder = opt.output_folder)
    compute_prf_metrics(pred_labels, gt_labels, output_folder = opt.output_folder)

    # print(dice_per_bits_level)
    # print(jaccard_per_bits_level)
    plot_latent_space(query_info, database_info, output_folder = opt.output_folder)
    plot_query_and_retrieved_img(query_imgs[:4], retrieved_imgs[:4],
                                 query_files[:4],retrieved_files[:4],
                                 output_folder = opt.output_folder)
    plot_query_and_retrieved_img(query_imgs[-6:], retrieved_imgs[-6:],
                                 query_files[-6:],retrieved_files[-6:],
                                 output_folder = opt.output_folder)
    plt.show()