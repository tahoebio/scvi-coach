import os
os.chdir('/vevo/drive_3/ANALYSIS/analysis_157/')

from glob import glob

import cudf
import cuml
from cuml.manifold import TSNE
import numpy as np
import pandas as pd
import plotnine as p
from pytorch_lightning.loggers import CSVLogger
from scipy import sparse
import scvi
import torch
import tqdm
import typer


def main():
    timestamp = '2024-12-29T03:43:35.279711'
    model_path = f'/vevo/drive_3/ANALYSIS/analysis_157/Data/scvi_models/{timestamp}/model.minified'

    model = scvi.model.SCVI.load(model_path)

    n_cells = f'{model.adata.shape[0]:,}'
    print(f'Fitting TSNE of {n_cells} cells')

    data_cudf = cudf.DataFrame.from_records(model.adata.obsm['_scvi_latent_qzm'])

    tsne = TSNE(verbose = True)
    embedding = tsne.fit_transform(data_cudf)
    embedding = pd.DataFrame(embedding.to_numpy(), index = model.adata.obs.index)
    embedding.columns = ['tsne_1', 'tsne_2']

    print(f'Saving TSNE coordines in {model_path} folder')    
    embedding.to_csv(f'{model_path}/tsne.csv')


if __name__ == "__main__":
    typer.run(main)