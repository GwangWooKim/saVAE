import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import scanpy as sc
import umap.umap_ as umap

from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

def preprocessing(adata, batch_key = None):
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy() # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata # freeze the state in `.raw`

    sc.pp.highly_variable_genes(
    adata,
    n_top_genes=1200,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key=batch_key
    )

def data_load(data_name, cov):

    covariate = torch.empty(0)
    batch = None

    if data_name == 'Two_moons':
        from sklearn.datasets import make_moons
        df, labels = make_moons(n_samples=4000, noise = 0.05, shuffle = False, random_state = 42)

    elif data_name == 'MNIST':
        from torchvision import datasets, transforms
        MNIST = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
        df = MNIST.data.view(60000,28*28).to(torch.float) / 255
        df = df.apply_(lambda x: 1 if x>= 0.5 else 0)
        labels = MNIST.targets
    
    elif data_name == 'cortex':
        import scvi
        adata = scvi.data.cortex()
        preprocessing(adata)
        df = adata.layers['counts']
        labels = adata.obs['cell_type']
    
    elif data_name == 'pbmc':
        import scvi
        adata = scvi.data.pbmc_dataset()
        preprocessing(adata)
        df = adata.layers['counts'].toarray()
        labels = adata.obs['str_labels']
    
    elif data_name == 'retina':
        import scvi
        from torch.nn import functional as F
        adata = scvi.data.retina()
        preprocessing(adata)
        df = adata.layers['counts']
        labels = adata.obs['labels']
        batch = LabelEncoder().fit_transform(adata.obs['batch'])
        if cov:
            covariate = F.one_hot(torch.Tensor(batch).to(torch.int64))

    elif data_name == 'heart_cell_atlas':
        import scvi
        from torch.nn import functional as F
        adata = scvi.data.heart_cell_atlas_subsampled()
        preprocessing(adata)
        df = adata.layers['counts'].toarray()
        labels = adata.obs['cell_type']
        if cov:
            donor = LabelEncoder().fit_transform(adata.obs['donor'])
            donor_one = F.one_hot(torch.Tensor(donor).to(torch.int64))
            cell_source = LabelEncoder().fit_transform(adata.obs['cell_source'])
            cell_source_one = F.one_hot(torch.Tensor(cell_source).to(torch.int64))
            batch = cell_source_one
            mito = torch.Tensor(adata.obs['percent_mito']).view(-1,1)
            ribo = torch.Tensor(adata.obs['percent_ribo']).view(-1,1)
            covariate = torch.concat((donor_one, cell_source_one, mito, ribo), dim=1)
        
    X = torch.Tensor(df)
    X_ = torch.concat((X,covariate), dim=1)
    labels_ = LabelEncoder().fit_transform(labels)

    return X, X_, labels, labels_, batch

def set_args(data_name):
    if data_name == 'Two_moons':
        dist = 'G'
        corr = cov = eval_ = False
        log_transform = False
        z_dim = 2
    
    elif data_name == 'MNIST':
        dist = 'B'
        corr = cov = eval_ = False
        log_transform = False
        z_dim = 10

    elif data_name == 'cortex':
        dist = 'NB'
        corr = cov = False
        log_transform = eval_ = True
        z_dim = 10

    elif data_name == 'pbmc':
        dist = 'NB'
        corr = cov = False
        log_transform = eval_ = True
        z_dim = 10
    
    elif data_name == 'retina':
        dist = 'NB'
        corr = cov = True
        log_transform = eval_ = True
        z_dim = 10

    elif data_name == 'heart_cell_atlas':
        dist = 'NB'
        corr = cov = True
        log_transform = eval_ = True
        z_dim = 10
    
    return dist, corr, cov, eval_, log_transform, z_dim

def seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    np.random.seed(num)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(num)

def make_simplices(graph, y=None):
    if y is not None:
      graph_ = umap.discrete_metric_simplicial_set_intersection(graph, y, far_dist = 5.0)
    else:
      graph_ = graph
    graph_ = graph.tocoo()
    graph_.sum_duplicates()

    # For smaller datasets we can use more epochs
    if graph_.shape[0] <= 10000:
        n_epochs = 500
    else:
        n_epochs = 200

    graph_.data[graph_.data < (graph_.data.max() / float(n_epochs))] = 0.0
    graph_.eliminate_zeros()

    head = graph_.row
    tail = graph_.col
    weight = graph_.data
    epochs_per_sample = umap.make_epochs_per_sample(graph_.data, n_epochs)

    return head, tail, weight, epochs_per_sample

def mp(function, object, num_cpu = cpu_count()):
    pool = Pool(num_cpu)
    return pool.map(function, object)

def make_tuple(tuple):
    i, head, tail, epochs_per_sample = np.array(tuple).T
    index = (head == i)
    return [tail[index], 
    epochs_per_sample[index], epochs_per_sample[index]]

def cal_graph(data, random_state = 42):
    graph = umap.UMAP(random_state = random_state, 
    min_dist = 0.0, 
    n_neighbors = 30,  
    transform_mode='graph',
    verbose=True).fit_transform(data)

    head, tail, weight, epochs_per_sample = make_simplices(graph, y=None)

    lst = [(i, head, tail, epochs_per_sample) for i in range(data.shape[0])]
    arr = np.array(mp(make_tuple, lst, num_cpu = int(cpu_count() / 2)), dtype=object)
    return arr

def check(tuple):
    data, labels_, i, j = np.array(tuple).T
    clustering = DBSCAN(min_samples = i+3, eps = 0.01 + j * 0.01).fit(data)
    ARI = metrics.adjusted_rand_score(labels_, clustering.labels_)
    NMI = metrics.normalized_mutual_info_score(labels_, clustering.labels_)
    return (ARI+NMI)/2

def DBSCAN_(data, labels_):
    lst = []
    for i in range(5):
        for j in range(200):
            lst.append((data, labels_, i,j))
    
    output = np.array(mp(check, lst, int(cpu_count()/4)))
    temp = lst[np.where(output==output.max())[0][-1]]
    i, j = temp[2], temp[3]
    clustering = DBSCAN(min_samples = i+3, eps = 0.01 + j * 0.01).fit(data)
    return clustering

def evaluation(data, labels_):
    dict_ = {}
    clustering = KMeans(random_state=42, n_clusters=len(set(labels_))).fit(data)
    ARI = metrics.adjusted_rand_score(labels_, clustering.labels_)
    NMI = metrics.normalized_mutual_info_score(labels_, clustering.labels_)
    dict_['KMeans'] = {'ARI': ARI, 'NMI': NMI, 'labels_': clustering.labels_}

    clustering = DBSCAN_(data, labels_)
    ARI = metrics.adjusted_rand_score(labels_, clustering.labels_)
    NMI = metrics.normalized_mutual_info_score(labels_, clustering.labels_)
    dict_['DBSCAN'] = {'ARI': ARI, 'NMI': NMI, 'labels_': clustering.labels_}
    return dict_
