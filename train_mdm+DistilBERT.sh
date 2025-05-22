python -m train.train_mdm \
--save_dir save/my_humanml_trans_dec_bert_512 \
--dataset humanml \
--diffusion_steps 50 \
--arch trans_dec \
--text_encoder_type bert \
--mask_frames \
--use_ema \
--device 0