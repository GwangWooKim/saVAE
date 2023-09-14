<img src="/imgs/saVAE.png" width="85%" height="85%">

# Similarity-assisted Variational Autoencoder (saVAE)
A similarity-assisted variational autoencoder (saVAE) is a new method that adopts similarity information in the framework of the VAE. If you want to know more details, please refer to [`saVAE_overvew.pdf`](https://github.com/GwangWooKim/saVAE/blob/main/saVAE_overview.pdf) in this repository that contains backgrounds, preliminaries, methods, results, and conclusions.

## Dependencies
* python==3.9.13
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
* `--covariate`: Whether to ues covariate information or not.
* `--correction`: Whether to do covariate correction or not (to improve the similarity table of the dataset).
* `--evaluation`: Whether to evaluate or not (on the inffered latent space of saVAE).   

The last three default values depend on the dataset.

### Description of the outputs

After training saVAE on the specified dataset, you will obtain the resulting files (dir: /output/data_name/). You can check them via `torch.load`.
* `df.pt`: The used training dataset.
* `df_.pt`: When the dataset contains some covariate information (`retina` or `heart_cell_atlas`), `df_.pt` is composed of `df.pt` and its covariate. Otherwise, it is the same as `df.pt`.
* `labels.pt`: The used original (string) labels.
* `labels_.pt`: The transformed labels via `sklearn.preprocessing.LabelEncoder`.
* `saVAE_latent.pt`: The encoded datapoints of `df.pt` from the ambient space to the learned latent one.
* `dict_.pt`: The evaluation results by using `saVAE_latent.pt` and `labels_.pt`.
* `saVAE_rec.pt`: In case of `Two_moons`, the reconstruected data is also saved.

### Visualization

One example to visualize the resulting output is as follows:

```python
import torch
import umap.umap_ as umap
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

# MNIST example
saVAE_latent = torch.load('saVAE_latent.pt')
saVAE_latent_2d = umap.UMAP(random_state=42, 
                            init=pca.fit_transform(saVAE_latent), 
                            n_epochs=1000
                            ).fit_transform(saVAE_latent)
labels_ = torch.load('labels_.pt')

fig, ax = plt.subplots(1,1)
temp = ax.scatter(saVAE_latent_2d[:, 0], 
                  saVAE_latent_2d[:, 1], 
                  s = 1.5,  
                  cmap='Spectral', 
                  c = labels_
                  )
plt.tick_params(top=False,
               bottom=False,
               left=False,
               right=False,
               labelleft=False,
               labelbottom=False)
plt.xlabel('UMAP_1')
plt.ylabel('UMAP_2')
plt.tight_layout()
```

<img src="/imgs/output.png" width="65%" height="65%">

Note that we don't have to do dimension reduction in case of `Two_moons`, because its latent dimension is 2.
