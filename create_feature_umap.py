import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt
model_name = 'vits14'
pretrained_model = True
train_type = 'pretrained'
feature_save_name = f'{model_name}_{train_type}'
dinov2_repo_dir = '/mnt/storage1/Haoran/projects/retina/retina-dinov2'
analysis_repo_dir = '/mnt/storage1/Haoran/projects/retina/retina-fundus'
# Read in the feature array
feature_array = np.load(f'{analysis_repo_dir}/feature/{feature_save_name}.npy')


# Create a 2D UMAP
umap_model = umap.UMAP(n_components=2)
umap_result = umap_model.fit_transform(feature_array)

# umap_result = pd.read_csv('/mnt/storage1/Haoran/projects/retina/src/Fundus_location_OCT_test_5-1-24_abbas_fundus_umap.csv', index_col=0)
# umap_result = np.load(f'{analysis_repo_dir}/umap/{feature_save_name}.npy')

# Plot the UMAP
plt.scatter(umap_result[:, 0], umap_result[:, 1])
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('2D UMAP')
plt.tight_layout()
plt.savefig(f'{analysis_repo_dir}/fig/{feature_save_name}.png', dpi=300)
plt.close()

# Save the UMAP coordinates
np.save(f'{analysis_repo_dir}/umap/{feature_save_name}.npy', umap_result)