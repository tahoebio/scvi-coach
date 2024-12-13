import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from glob import glob
from datetime import datetime

import anndata
from anndata.experimental.multi_files import AnnCollection
import numpy as np
from pytorch_lightning.loggers import CSVLogger
from scipy import sparse
import scvi
from scvi.train import SaveCheckpoint
import torch
import tqdm
import typer

from coach import CollectionAdapter

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # Create an ISO timestamp
    timestamp = datetime.now().isoformat()
    print(f'Training model version {timestamp}')

    base_dir = f'/vevo/drive_3/ANALYSIS/analysis_157/Data/scvi_models/{timestamp}'
    print(f'Base directory: {base_dir}')

    # Load the data
    pattern = '/atlas/p*/*.h5ad'
    files = glob(pattern)

    adatas = []
    for fname in files:
        ad_ = anndata.read_h5ad(fname, backed = 'r')
        ad_.layers['counts'] = ad_.X
        adatas += [ad_]  

    anco = AnnCollection(adatas, label = 'dataset')

    print(anco)

    collection_adapter = CollectionAdapter(anco)

    torch.set_float32_matmul_precision('high')
    scvi.settings.dl_num_workers = 8
    scvi.settings.dl_persistent_workers = False
    scvi.settings.num_threads = 8

    # Set up model
    scvi.model.SCVI.setup_anndata(
        collection_adapter,
        layer = 'counts',
        size_factor_key = 'tscp_count'
    )
    model = scvi.model.SCVI(collection_adapter, gene_likelihood = 'nb')

    # Training settings
    batch_size = 128

    # How much to log per epoch
    val_check_interval = 0.25
    train_log_interval = 0.05

    steps_per_epoch = (collection_adapter.shape[0] // batch_size)
    every_n_steps = int(steps_per_epoch * train_log_interval)

    checkpoint_callback = SaveCheckpoint(
        dirpath = f'{base_dir}/checkpoints/',
        filename = 'model_epoch_{epoch}',
        monitor = 'validation_loss'
    )

    csv_logger = CSVLogger(save_dir = base_dir + '/scvi_logs/', name = f'train')

    # Run training
    model.train(
        enable_progress_bar = True,
        batch_size = batch_size,
        max_epochs = 10,
        accelerator = 'cuda',
        devices = [0],
        logger = csv_logger,
        log_every_n_steps = every_n_steps,
        check_val_every_n_epoch = 1,
        val_check_interval = val_check_interval,
        callbacks = [checkpoint_callback]
    )


    # Generate cell representations
    print('Training finished, creating representations...')
    qzm, qzv = model.get_latent_representation(give_mean = False, return_dist = True)

    # Manually create minified AnnData
    all_zeros = sparse.csr_matrix(model.adata.shape)

    new_adata = anndata.AnnData(X = all_zeros, obs = model.adata.obs, var = adatas[0].var)
    new_adata.layers['counts'] = all_zeros

    new_adata = model._validate_anndata(new_adata)

    use_latent_qzm_key = 'X_latent_qzm'
    use_latent_qzv_key = 'X_latent_qzv'

    new_adata.obsm[use_latent_qzm_key] = qzm
    new_adata.obsm[use_latent_qzv_key] = qzv

    _ADATA_MINIFY_TYPE = scvi.data._constants.ADATA_MINIFY_TYPE.LATENT_POSTERIOR

    mini_fields = model._get_fields_for_adata_minification(_ADATA_MINIFY_TYPE)

    _LATENT_QZM_KEY = mini_fields[0].attr_key
    _LATENT_QZV_KEY = mini_fields[1].attr_key
    _OBSERVED_LIB_SIZE_KEY = mini_fields[2].attr_key
    _ADATA_MINIFY_TYPE_UNS_KEY = mini_fields[3].attr_key

    new_adata.uns[_ADATA_MINIFY_TYPE_UNS_KEY] = _ADATA_MINIFY_TYPE
    new_adata.obsm[_LATENT_QZM_KEY] = new_adata.obsm[use_latent_qzm_key]
    new_adata.obsm[_LATENT_QZV_KEY] = new_adata.obsm[use_latent_qzv_key]
    new_adata.obs[_OBSERVED_LIB_SIZE_KEY] = new_adata.obs['tscp_count']

    del new_adata.uns['_scvi_uuid']

    model._update_adata_and_manager_post_minification(new_adata, _ADATA_MINIFY_TYPE)
    model.module.minified_data_type = _ADATA_MINIFY_TYPE

    print(model)
    print(model.adata)

    # Save model and minified AnnData
    model.save(base_dir + '/model.minified', save_anndata = True, overwrite = True)

    print(f'Model saved at {base_dir}/model.minified')


if __name__ == "__main__":
    typer.run(main)

