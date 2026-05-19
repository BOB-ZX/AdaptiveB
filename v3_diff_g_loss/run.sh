CUDA_VISIBLE_DEVICES=1 python main.py fit --config config.yaml --trainer.logger.name brast_exp  \
    --data.dataset_dir /home/zxl/data/SelfRDB/data/BRATS2021_processed --data.source_modality t1  --data.target_modality t2 \
    --data.train_batch_size 2 --data.val_batch_size 2 --trainer.max_epoch 100  --trainer.devices 0,