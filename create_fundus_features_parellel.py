import torch
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import ssl
import gc
from dinov2.models.vision_transformer import vit_base, vit_large
import glob
from multiprocessing import Process, set_start_method

# Set the start method to 'spawn'
set_start_method('spawn', force=True)

# Constants
repo_dir = '/cnvrg'
model_name = 'vitb16_scratch_lr_5e-04'
model_arch = model_name.split('_')[0]
pretrained_model = False
train_type = 'pretrained' if pretrained_model else 'scratch'

# Helper functions
def list_files(dataset_path):
    print("Listing files in:", dataset_path)
    images = []
    for root, _, files in sorted(os.walk(dataset_path)):
        for name in sorted(files):
            if name.lower().endswith('.tif'):
                images.append(os.path.join(root, name))
    print(f"Found {len(images)} .tif files.")
    return images

class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = list_files(self.img_dir)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

    def get_file_names(self):
        file_names = [os.path.basename(path) for path in self.images]
        print(f"Extracted {len(file_names)} file names.")
        return file_names

# Example usage
dir_path = "/data/fundus"
dataset = CustomImageDataset(dir_path)
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

file_names = np.array(dataset.get_file_names())

# Model selection
if model_arch == 'vitb16':
    dinov2_model = vit_base(patch_size=16, img_size=224, init_values=1.0, block_chunks=0)
elif model_arch == 'vitl16':
    dinov2_model = vit_large(patch_size=16, img_size=224, init_values=1.0, block_chunks=0)

# Function to modify keys of pretrained weights
def modify_keys(pretrained_weights):
    new_weights = {}
    for k, v in pretrained_weights.items():
        if k.startswith('backbone.'):
            k = k.replace('backbone.', '')
        parts = k.split('.')
        if len(parts) > 1 and parts[0] == 'blocks':
            k = '.'.join(parts[:1] + parts[2:])
        if parts[0] != 'dino_head':
            new_weights[k] = v
    return new_weights

# Load model weights
ssl._create_default_https_context = ssl._create_unverified_context

def create_features(train_dataloader, dinov2_model, device):
    final_img_features = []
    final_img_filepaths = []

    for image_tensors, file_paths in tqdm(train_dataloader):
        try:
            with torch.no_grad():
                img_t = image_tensors.to(device)
                image_features = dinov2_model(img_t)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().tolist()
                final_img_features.extend(image_features)
                final_img_filepaths.extend(list(file_paths))
            del img_t, image_features
            torch.cuda.empty_cache()
        except Exception as e:
            print("Exception occurred:", e)
            break
        finally:
            gc.collect()
    return final_img_features

def process_job(training_folders, gpu_id):
    for training_folder in training_folders:
        checkpoint_name = os.path.basename(training_folder)
        checkpoint_path = os.path.join(training_folder, 'teacher_checkpoint.pth')
        pretrained_weights = torch.load(checkpoint_path)['teacher']
        pretrained_weights = modify_keys(pretrained_weights)

        # Load model
        if model_arch == 'vitb16':
            dinov2_model = vit_base(patch_size=16, img_size=224, init_values=1.0, block_chunks=0)
        elif model_arch == 'vitl16':
            dinov2_model = vit_large(patch_size=16, img_size=224, init_values=1.0, block_chunks=0)
        
        dinov2_model.load_state_dict(pretrained_weights)

        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using device: cuda:{gpu_id}")
        dinov2_model.to(device)

        # Feature extraction
        final_img_features = create_features(train_dataloader, dinov2_model, device)

        # Save features
        final_img_features_array = np.array(final_img_features)
        feature_dir = f'{repo_dir}/feature/{model_name}'
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        np.save(f'{feature_dir}/{checkpoint_name}.npy', final_img_features_array)

if __name__ == "__main__":
    model_path = f'{repo_dir}/result_{model_name}/eval'
    training_folders = sorted(glob.glob(f'{model_path}/training_*'))
    print(training_folders)

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # Distribute training folders among GPUs
    folders_per_gpu = len(training_folders) // num_gpus
    processes = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * folders_per_gpu
        end_idx = start_idx + folders_per_gpu
        if gpu_id == num_gpus - 1:  # Last GPU takes any remaining folders
            end_idx = len(training_folders)
        gpu_folders = training_folders[start_idx:end_idx]
        
        p = Process(target=process_job, args=(gpu_folders, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
