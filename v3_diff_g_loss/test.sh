CUDA_VISIBLE_DEVICES=0 python main.py test --config config.yaml --trainer.logger.name brast_exp  --data.dataset_dir /home/zxl/data/SelfRDB/data/BRATS2021_processed_new \
        --data.source_modality t1  --data.target_modality t2 --data.test_batch_size 32 \
        --ckpt_path  /home/zxl/data/SelfRDB_ALL/SelfRDB14/v3_diff_g_loss/logs/brast_exp/version_1/checkpoints/epoch=64-step=81250.ckpt