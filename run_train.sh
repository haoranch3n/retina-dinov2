export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 train_individual_worker_pretrained.py --config-file dinov2/configs/train/vitb14.yaml --output-dir /cnvrg/result train.dataset_path=Fundus:root=/data/fundus

