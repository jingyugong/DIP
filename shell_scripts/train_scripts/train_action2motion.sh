python -m train.train_mdm --dataset mixed --save_dir save/trained_models/mixed_action2motion --cond_mask_prob 0 --lambda_vel 1 --num_frames 160 --overwrite
#python -m train.train_mdm --save_dir save/trained_models/humanact12 --dataset humanact12 --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1 --overwrite --num_steps 100000
