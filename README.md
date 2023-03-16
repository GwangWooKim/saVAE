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



