import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.manifold import TSNE
import os 

model_name = 'vitb16_scratch_lr_5e-04'
# dinov2_repo_dir = '/mnt/storage1/Haoran/projects/retina/retina-dinov2'
dinov2_repo_dir = '/cnvrg'
# analysis_repo_dir = '/mnt/storage1/Haoran/projects/retina/retina-fundus'
# Read in the feature array
feature_dir = f'{dinov2_repo_dir}/feature/{model_name}/original'

checkpoints = sorted(glob.glob(f'{feature_dir}/training*.npy'))

def create_umap(features, numComponents, save_name):
    # Create a 2D UMAP
    umap_model = umap.UMAP(n_components=numComponents)
    umap_result = umap_model.fit_transform(features)
    if not os.path.exists(f'{dinov2_repo_dir}/feature/umap-{numComponents}'):
        os.makedirs(f'{dinov2_repo_dir}/feature/umap-{numComponents}')
    np.save(f'{dinov2_repo_dir}/feature/umap-{numComponents}/{save_name}', umap_result)

def create_tsne(features, numComponents, save_name):
    # Create a 2D t-SNE
    tsne_model = TSNE(n_components=numComponents)
    tsne_result = tsne_model.fit_transform(features)
    if not os.path.exists(f'{dinov2_repo_dir}/feature/tsne-{numComponents}'):
        os.makedirs(f'{dinov2_repo_dir}/feature/tsne-{numComponents}')
    np.save(f'{dinov2_repo_dir}/feature/tsne-{numComponents}/{save_name}', tsne_result)

for checkpoint in checkpoints:
    print(checkpoint)
    feature_array = np.load(checkpoint)
    feature_save_name = os.path.basename(checkpoint[:-4]) + '.npy'
    create_umap(feature_array, 2, feature_save_name)
    create_umap(feature_array, 20, feature_save_name)
    create_tsne(feature_array, 2, feature_save_name)

