CUDA_VISIBLE_DEVICES=3 python main.py test --config config.yaml --trainer.logger.name brast_exp  --data.dataset_dir /data/8T-0/zxl/datasets/BRATS2021_processed \
        --data.source_modality t1  --data.target_modality t2 --data.test_batch_size 64 \
        --ckpt_path  /data/8T-0/zxl/DualB/v3/logs/brast_exp/version_0/checkpoints/epoch=19-step=3120.ckpt