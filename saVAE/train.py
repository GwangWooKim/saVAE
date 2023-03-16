import torch
import numpy as np
import umap.umap_ as umap

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_bce = torch.nn.BCELoss(reduction='none')

def nb_llik(x, mean, inv_disp):
  return (x * torch.log(mean / inv_disp) -
          x * torch.log(1 + mean / inv_disp) -
          inv_disp * torch.log(1 + mean / inv_disp) +
          torch.lgamma(x + inv_disp) -
          torch.lgamma(inv_disp) -
          torch.lgamma(x + 1))

def cal_reconstruction_error(x, x_hat, model_para, dist):
    if dist == 'B':
        return loss_bce(x_hat, x).sum()
    elif dist == 'NB':
        return -nb_llik(x, x_hat.clamp(1e-10), model_para.exp().clamp(1e-10)).sum()
    elif dist == 'G':
        return torch.sum((x - x_hat)**2)

def train_VAE(x, z_mu, z_logvar, x_hat, model_para, dist, beta):
    RE = cal_reconstruction_error(x, x_hat, model_para, dist)
    KL = - 0.5 * torch.sum(1 + z_logvar - z_mu**2 - z_logvar.exp(), dim=1).sum()
    LOSS = (RE + beta * KL) / x.size(0)
    return LOSS, RE, KL

def train_corr(model, X_, batch_size, epochs):
    data_loader = torch.utils.data.DataLoader(X_, batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_dim = model.x_dim
    dist = model.dist

    RE_lst = []
    KL_lst = []
    VAE_lst = []

    for epoch in tqdm(range(1,  epochs + 1)):
        RE_ = []
        KL_ = []
        VAE_ = []

        for i, x in enumerate(data_loader):
            optimizer.zero_grad()
            x = x.to(device)
            z_mu, z_logvar, z, x_hat = model(x)

            VAE, RE, KL = train_VAE(x[:, :x_dim], z_mu, z_logvar, x_hat, model.para, dist, 1)
            SUM = VAE
            SUM.backward()

            RE_.append(RE.item())
            KL_.append(KL.item())
            VAE_.append(VAE.item())

            optimizer.step()
        
        RE_lst.append(np.array(RE_).mean())
        KL_lst.append(np.array(KL_).mean())
        VAE_lst.append(np.array(VAE_).mean())
    
    return model, RE_lst, KL_lst, VAE_lst

def sampling(arr_, indices, epoch, X):
    targets, epochs_per_sample, epoch_of_next_sample = arr_[indices, :].T
    lsts = list(map(lambda x: x <= epoch, epoch_of_next_sample))
    arr_[indices, 2] += lsts * epochs_per_sample 
    temp = sum([target[lst].tolist() for target, lst in zip(targets, lsts)], [])
    sample_edge_to_x = torch.repeat_interleave(X[indices, :], torch.Tensor(list(map(lambda x: x.sum(), lsts))).to(torch.int), dim=0)
    sample_edge_from_x = X[temp, :]
    return sample_edge_to_x, sample_edge_from_x

def convert_distance_to_probability(distances, a=1.0, b=1.0):
  return 1.0 / (1.0 + a * distances ** (2 * b))

def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    # cross entropy
    attraction_term = -probabilities_graph * torch.log(
        torch.clamp(probabilities_distance, EPS, 1.0)
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * torch.log(torch.clamp(1.0 - probabilities_distance, EPS, 1.0))
        * repulsion_strength
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE

def cal_umap_loss(embedding_to, embedding_from, embedding_neg_to,
embedding_neg_from, batch_size, negative_sample_rate, _a, _b,):
  distance_embedding = torch.concat(
  [
      torch.norm(embedding_to - embedding_from, dim=1),
      torch.norm(embedding_neg_to - embedding_neg_from, dim=1),
  ],
  dim=0,
  )   

  probabilities_graph = torch.concat(
  [torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)], dim=0,
  ).to(device)

  probabilities_distance = convert_distance_to_probability(
    distance_embedding, _a, _b
  )

  (_, _, ce_loss) = compute_cross_entropy(
      probabilities_graph,
      probabilities_distance,
  )
  return ce_loss.mean()

def train_UMAP(embedding_to, embedding_from, _a, _b, negative_sample_rate = 5):
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    index = torch.randint(high = len(repeat_neg), size = (len(repeat_neg),))
    embedding_neg_from = torch.index_select(repeat_neg, 0, index.to(device))
    UMAP = cal_umap_loss(embedding_to, embedding_from, embedding_neg_to,
        embedding_neg_from, embedding_to.size(0), negative_sample_rate, _a, _b)
    return UMAP

def train_main(model, X, X_, arr, batch_size, epochs):
    index_loader = torch.utils.data.DataLoader(range(X_.size(0)), batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_dim = model.x_dim
    dist = model.dist
    w = model.weight
    u = model.update_ratio
    _a, _b = umap.find_ab_params(1.0, min_dist=0.0)

    RE_lst = []
    KL_lst = []
    VAE_lst = []
    UMAP_lst = []
    SUM_lst = []

    for epoch in tqdm(range(1,  epochs + 1)):
        RE_ = []
        KL_ = []
        VAE_ = []
        UMAP_ = []
        SUM_ = []

        for i, indices in enumerate(index_loader):
            optimizer.zero_grad()

            sample_edge_to_x, sample_edge_from_x = sampling(arr, indices, epoch, X_)
            embedding_to, _, _ = model.encode(sample_edge_to_x[:, :x_dim].to(device))
            embedding_from, _, _ = model.encode(sample_edge_from_x[:, :x_dim].to(device))

            UMAP = train_UMAP(embedding_to, embedding_from, _a, _b)
        
            if i % u == 0:
                x = torch.concat((sample_edge_to_x, sample_edge_from_x), dim=0).to(device)
                z_mu, z_logvar, z, x_hat = model(x)

                VAE, RE, KL = train_VAE(x[:, :x_dim], z_mu, z_logvar, x_hat, model.para, dist, 1)
                SUM = w * VAE + UMAP
                SUM.backward()

                RE_.append(RE.item())
                KL_.append(KL.item())
                VAE_.append(VAE.item())
                UMAP_.append(UMAP.item())
                SUM_.append(SUM.item())
            else:
                UMAP.backward()
            
            optimizer.step()

        RE_lst.append(np.array(RE_).mean())
        KL_lst.append(np.array(KL_).mean())
        VAE_lst.append(np.array(VAE_).mean())
        UMAP_lst.append(np.array(UMAP_).mean())
        SUM_lst.append(np.array(SUM_).mean())

    return model, VAE_lst, SUM_lst