export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 train_individual_worker.py --config-file dinov2/configs/train/vitb16_short.yaml --output-dir /cnvrg/result train.dataset_path=Fundus:root=/data/fundus

