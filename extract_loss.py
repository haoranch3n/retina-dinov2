import re
import pandas as pd
import os
import json

# Path to your log file
# repo_dir = '/mnt/storage1/Haoran/projects/retina/retina-dinov2'
repo_dir = '/cnvrg'
# model_name = 'vitb16_scratch_lr_5e-04'
# model_name = 'vitb14_pretrained_lr_5e-6'
# model_name = 'vitb16-scratch-fundus-batch-7-batchsize-64_lr-0.0005'
model_name = 'vitb16-scratch-fundus-batch-7-batchsize-64_lr-0.0005'
# model_name = 'vitb16-scratch-fundus-batch-7-batchsize-32-1'
# model_name = 'vitb16-scratch-fundus-batch-7-batchsize-128-1'
log_file_path = f'{repo_dir}/result/training_metrics.json'

def read_multiple_json_objects(file_path):
    with open(file_path, 'r') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    return data

# log_file_path = 'your_log_file_path.json'
data = read_multiple_json_objects(log_file_path)
# data = data[62502:]
# # Read the log file and extract the relevant information
# with open(log_file_path, 'r') as file:
#     for line in file:
#         iteration = int(line['iteration'])
#         total_loss = float(line['total_loss'])
#         dino_local_crops_loss = float(line['dino_local_crops_loss'])
#         dino_global_crops_loss = float(line['dino_global_crops_loss'])
#         koleo_loss = float(line['koleo_loss'])
#         ibot_loss = float(line['ibot_loss'])
        
#         data.append({
#             'iteration': iteration,
#             'total_loss': total_loss,
#             'dino_local_crops_loss': dino_local_crops_loss,
#             'dino_global_crops_loss': dino_global_crops_loss,
#             'koleo_loss': koleo_loss,
#             'ibot_loss': ibot_loss
#         })

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Optionally, save the DataFrame to a CSV file
# if not os.path.exists(f'{repo_dir}/loss/{model_name}'):
    # os.makedirs(f'{repo_dir}/loss/{model_name}')
# df.to_csv(f'{repo_dir}/loss/{model_name}/extracted_losses.csv', index=False)

print(df['total_loss'].min())
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame with the extracted data
# If you need to load the data from a CSV, uncomment the next line
# df = pd.read_csv('extracted_losses.csv')

# Create subplots
fig, axs = plt.subplots(6, 1, figsize=(10, 23), sharex=True)

# Plot each loss on a separate subplot
axs[0].plot(df['iteration'], df['total_loss'], label='Total Loss', color='blue')
axs[0].set_ylabel('Total Loss')
axs[0].legend()

axs[1].plot(df['iteration'], df['dino_local_crops_loss'], label='Dino Local Crops Loss', color='orange')
axs[1].set_ylabel('Dino Local Crops Loss')
axs[1].legend()

axs[2].plot(df['iteration'], df['dino_global_crops_loss'], label='Dino Global Crops Loss', color='green')
axs[2].set_ylabel('Dino Global Crops Loss')
axs[2].legend()

axs[3].plot(df['iteration'], df['koleo_loss'], label='Koleo Loss', color='red')
axs[3].set_ylabel('Koleo Loss')
axs[3].legend()

axs[4].plot(df['iteration'], df['ibot_loss'], label='Ibot Loss', color='purple')
axs[4].set_ylabel('Ibot Loss')
axs[4].set_xlabel('Iteration')
axs[4].legend()

axs[5].plot(df['iteration'], df['lr'], label='Learning Rate', color='cyan')
axs[5].set_ylabel('Learning Rate')
axs[5].set_xlabel('Iteration')
axs[5].legend()

# Adjust layout
plt.tight_layout()
plt.savefig(f'{repo_dir}/result/extracted_losses.png', dpi=300)
plt.close()


