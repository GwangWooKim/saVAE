from utils import *
from model import *
from train import *

import argparse
import os

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", default = 42)
parser.add_argument("-e", "--epochs", default = 200)
parser.add_argument("-b", "--batch_size", default = 128)
parser.add_argument("-w", "--weight", default = 1e-3)
parser.add_argument("-r", "--update_ratio", default = 5)
parser.add_argument("-d", "--data", choices = ['Two_moons' , 'MNIST', 'cortex', 'pbmc', 'retina', 'heart_cell_atlas'], default = 'Two_moons')
parser.add_argument('--correction', dest='corr', action=argparse.BooleanOptionalAction)
parser.add_argument('--covariate', dest='cov', action=argparse.BooleanOptionalAction)
parser.add_argument('--evaluation', dest='eval', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dist, corr, cov, eval_, log_transform, z_dim = set_args(args.data)
    arr = None

    if args.corr == None:
        args.corr = corr

    if args.cov == None:
        args.cov = cov
    
    if args.eval == None:
        args.eval = eval_
    
    X, X_, labels, labels_, batch = data_load(args.data, args.cov)
    x_dim = X.size(1)
    n_cov = X_.size(1) - X.size(1)

    name = 'Data_' + args.data + '_corr_' + str(args.corr) + '_cov_' + str(args.cov)
    try:
        os.makedirs(f'./output/{name}')
    except:
        pass

    if args.corr:
        if args.data == 'Two_moons':
            # In this case, model_corr will be just a vanilla VAE which does not correct anything.
            # Instead, it demonstrates that the vanilla VAE does not produce a good representation of the generated data.
            model_corr = VAE(
                x_dim = x_dim,
                h_dim = 20,
                z_dim = z_dim,
                dist = dist,
                log_transform = log_transform,
                n_cov = n_cov
            ).to(device)
        else:
            model_corr = VAE_corr(
                x_dim = x_dim,
                h_dim = 128,
                z_dim = z_dim,
                dist = dist,
                log_transform = log_transform,
                n_cov = n_cov
            ).to(device)

        seed(args.seed)
        model_corr, RE_lst, KL_lst, VAE_lst = train_corr(model = model_corr, 
                                                        X_ = X_, 
                                                        batch_size = args.batch_size, 
                                                        epochs = args.epochs)
            
        model_corr.eval()
        z_mu, _, _, x_hat = model_corr(X_.to(device))
        latent = z_mu.cpu().detach().numpy()
        x_hat = x_hat.cpu().detach().numpy()

        if args.data == 'Two_moons':
            torch.save(latent, f'./output/{name}/VAE_latent.pt')
            torch.save(x_hat, f'./output/{name}/VAE_rec.pt')
            torch.save(labels, f'./output/{name}/labels.pt')
            torch.save(labels_, f'./output/{name}/labels_.pt')
        else:
            arr = cal_graph(latent)
    
    if arr is None:
        arr = cal_graph(torch.log(1+X) if log_transform else X)
    
    model = VAE(
                x_dim = x_dim,
                h_dim = 20 if args.data == 'Two_moons' else 512,
                z_dim = z_dim,
                dist = dist,
                log_transform = log_transform,
                n_cov = n_cov,
                weight = args.weight,
                update_ratio = args.update_ratio
            ).to(device)
    
    seed(args.seed)
    model, VAE_lst, SUM_lst = train_main(model = model, 
                                                X = X,
                                                X_ = X_, 
                                                arr = arr,
                                                batch_size = args.batch_size, 
                                                epochs = args.epochs)
    
    model.eval()
    z_mu, _, _, x_hat = model(X_.to(device))
    latent = z_mu.cpu().detach().numpy()
    x_hat = x_hat.cpu().detach().numpy()

    torch.save(X, f'./output/{name}/df.pt')
    torch.save(X_, f'./output/{name}/df_.pt')
    torch.save(labels, f'./output/{name}/labels.pt')
    torch.save(labels_, f'./output/{name}/labels_.pt')
    torch.save(latent, f'./output/{name}/saVAE_latent.pt')
    
    if args.data == 'Two_moons':
        torch.save(x_hat, f'./output/{name}/saVAE_rec.pt')

    if args.eval:
        dict_ = evaluation(latent, labels_)
        torch.save(dict_, f'./output/{name}/dict_.pt')
    
    print('Done')

if __name__ == '__main__':
    main()
