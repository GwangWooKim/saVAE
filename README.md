<img src="/imgs/saVAE.png" width="75%" height="75%">

# Similarity-assisted Variational Autoencoder (saVAE)
A similarity-assisted variational autoencoder (saVAE) is a new method that adopts similarity information in the framework of the VAE. 

## Dependencies
* torch==1.13.1
* scanpy==1.9.3
* loompy==3.0.7
* scvi-tools==0.20.2
* umap-learn==0.5.3

Other dependencies can be found at `requirement.txt`

## How to use
All arguments of the implementation are set to reproduce the results of the paper. It is enough to specify the data name. The available datasets are `Two_moons`, `MNIST`, `cortex`, `pbmc`, `retina`, and `heart_cell_atlas`.

### Example
    $ python main.py -d Two_moons
* `--data` (or `-d`): The data name. `Default = Two_moons`
* `--epochs` (or `-e`): The number of iterations for training. `Default = 200`
* `--batch_size` (or `-b`): The number of mini-batch. `Default = 128`
* `--weight` (or `-w`): The weight between VAE and UMAP in saVAE. `Default = 1e-3`
* `--updata_ratio` (or `-r`): The number of UMAP iterations per VAE iteration. `Default = 5`
* `--correction`: Whether the covariate correction does or not in order to improve the similarity table of the dataset.
* `--covariate`: Whether the covariate uses or not.
* `--evaluation`: Whether the evaluation does or not (on the inffered latent space of saVAE).   

The last three default values depend on the dataset.

### The resulting files

After training saVAE on the specified dataset, you will obtain 


