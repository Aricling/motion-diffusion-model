import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.BERT.BERT_encoder import load_bert
from utils.misc import WeightedSum
from clip.utils.lora_util import apply_lora_attn_mlp, init_finetuned_clip_and_freeze
import z_config
from clip.utils.add_lora import prepare_lora_clip_model


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)
        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.mask_frames = kargs.get('mask_frames', False)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.emb_policy = kargs.get('emb_policy', 'add')

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=kargs.get('pos_embed_max_len', 5000))
        self.emb_trans_dec = emb_trans_dec

        self.pred_len = kargs.get('pred_len', 0)
        self.context_len = kargs.get('context_len', 0)
        self.total_len = self.pred_len + self.context_len
        self.is_prefix_comp = self.total_len > 0
        self.all_goal_joint_names = kargs.get('all_goal_joint_names', [])
        
        self.multi_target_cond = kargs.get('multi_target_cond', False)
        self.multi_encoder_type = kargs.get('multi_encoder_type', 'multi')
        self.target_enc_layers = kargs.get('target_enc_layers', 1)
        if self.multi_target_cond:
            if self.multi_encoder_type == 'multi':
                self.embed_target_cond = EmbedTargetLocMulti(self.all_goal_joint_names, self.latent_dim)
            elif self.multi_encoder_type == 'single':
               self.embed_target_cond = EmbedTargetLocSingle(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)       
            elif self.multi_encoder_type == 'split':
               self.embed_target_cond = EmbedTargetLocSplit(self.all_goal_joint_names, self.latent_dim, self.target_enc_layers)     
        
        if z_config.get_diy_config().training.split_info_injection:
            half_layers = self.num_layers // 2

            print(f"Split mode enabled. Using half layers: {half_layers} for both encoder and decoder.")

            # 初始化前半部分 encoder 层 (half_layers 层)
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation
            )
            self.encoder_layers = nn.ModuleList([
                seqTransEncoderLayer for _ in range(half_layers)
            ])

            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation  # 注意你原来写的是 activation，这里保持一致，不过最好也用 self.activation
            )
            self.decoder_layers = nn.ModuleList([
                seqTransDecoderLayer for _ in range(half_layers)
            ])

            # 不再初始化完整的 self.seqTransEncoder 或 self.seqTransDecoder
            # 而是在 forward 中根据 version 手动控制 encoder 和 decoder layers 的执行顺序

            # 标记当前为 split 模式
            self.split_info_mode = True
            self.split_version = z_config.get_diy_config().training.split_info_version

        else:
            if self.arch == 'trans_enc':
                print("TRANS_ENC init")
                seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                            num_layers=self.num_layers)
            elif self.arch == 'trans_dec':
                print("TRANS_DEC init")
                seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=activation)
                self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                            num_layers=self.num_layers)
            elif self.arch == 'gru':
                print("GRU init")
                self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
            else:
                raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if z_config.get_diy_config().model.use_contronet_injection:
            print("======initialize controlnet=======")
            self.input_motion_token_layer=nn.Linear(self.latent_dim, self.latent_dim)
            self.c_input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
            self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            c_seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

            self.c_seqTransEncoder = nn.TransformerEncoder(c_seqTransEncoderLayer,
                                                        num_layers=self.num_layers)
            
            self.zero_convs = self.zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]))
            
            self.c_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

            self.c_joint_integration_layer = nn.Linear(in_features=7, out_features=1)

            if self.cond_mode != 'no_cond':
                if 'text' in self.cond_mode:
                    self.c_embed_text = nn.Linear(self.clip_dim, self.latent_dim)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                # We support CLIP encoder and DistilBERT
                print('EMBED TEXT')
                
                self.text_encoder_type = kargs.get('text_encoder_type', 'clip')
                
                if self.text_encoder_type == "clip":
                    print('Loading ori CLIP...')
                    self.clip_version = clip_version
                    self.clip_model_ori = self.load_ori_clip(clip_version)
                    for param in self.clip_model_ori.parameters():
                        param.requires_grad = False     ## 参数验证已通过

                    print('Loading lora CLIP...')
                    ori_clip_model = self.load_ori_clip(clip_version)
                    self.lra_clip_model = prepare_lora_clip_model(
                        clip_model=ori_clip_model,
                        use_lora=True,
                        lora_mlp=True,
                        lora_attn=True,
                        num_new_tokens=3,
                        save_dir=kargs.get("save_dir", None),
                    )

                elif self.text_encoder_type == 'bert':
                    assert self.arch == 'trans_dec'
                    # assert self.emb_trans_dec == False # passing just the time embed so it's fine
                    print("Loading BERT...")
                    # bert_model_path = 'model/BERT/distilbert-base-uncased'
                    bert_model_path = 'distilbert/distilbert-base-uncased'
                    self.clip_model = load_bert(bert_model_path)  # Sorry for that, the naming is for backward compatibility
                    self.encode_text = self.bert_encode_text
                    self.clip_dim = 768
                else:
                    raise ValueError('We only support [CLIP, BERT] text encoders') 
                
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

        if z_config.get_diy_config().training_input.use_end2end_ding_training:
            ## 这里相当于第一个实验和第二个实验
            if z_config.get_diy_config().training_input.use_cls_token:
                self.ori_clip_proj = nn.Linear(self.clip_dim, self.latent_dim)
                self.lra_clip_proj = nn.Linear(self.clip_dim, self.latent_dim)
                if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                    self.MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
                    self.lora_clip_MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
            if z_config.get_diy_config().training_input.not_use_cls_token:
                if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                    self.MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
                    self.lora_clip_MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
            
            ## 这里是第三个实验
            if z_config.get_diy_config().training_input.use_mean_MB_as_cls_in_gt_branch:
                self.mean_motion_bert_proj = nn.Linear(self.clip_dim, self.latent_dim)
                self.lra_clip_proj = nn.Linear(self.clip_dim, self.latent_dim)
                if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                    self.MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
                    self.lora_clip_MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)

            ## 这里是第四个实验
            if z_config.get_diy_config().training_input.both_use_ori_clip_cls_token:
                self.ori_clip_proj = nn.Linear(self.clip_dim, self.latent_dim)
                if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                    self.MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
                    self.lora_clip_MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            nn.init.zeros_(p)
        return module
    
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_ori_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def lora_clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 75 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length <= default_context_length
            texts, texts_tokens = clip.tokenize(raw_text, context_length=context_length, truncate=True) # [bs, context_length] # if n_tokens > context_length -> will truncate
            texts_lens_list = [len(text_token) for text_token in texts_tokens]
            texts_tokens_padded=torch.zeros([texts.shape[0], default_context_length], dtype=texts.dtype, device=texts.device)
            for i, text_tokens in enumerate(texts_tokens):
                texts_tokens_padded[i, :texts_lens_list[i]] = torch.tensor(text_tokens)

        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate

        texts = texts.to(device)
        texts_tokens_padded = texts_tokens_padded.to(device)
        
        if z_config.get_diy_config().model.use_avg_clip_model:  ## 在这里lora是要被训练的，根本就没有avg这种定义
            encoded_text = self.lra_clip_model.lora_encode_text(texts, texts_lens_list=texts_lens_list).float()    ## 这里的texts_lens_list只是为了之后提取index更加方便
        else:
            encoded_text = self.lra_clip_model.lora_encode_text(texts, texts_lens_list=texts_lens_list).float()

        return encoded_text, texts_lens_list
    
    def ori_clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 75 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length <= default_context_length
            texts, texts_tokens = clip.tokenize(raw_text, context_length=context_length, truncate=True) # [bs, context_length] # if n_tokens > context_length -> will truncate
            texts_lens_list = [len(text_token) for text_token in texts_tokens]
            texts_tokens_padded=torch.zeros([texts.shape[0], default_context_length], dtype=texts.dtype, device=texts.device)
            for i, text_tokens in enumerate(texts_tokens):
                texts_tokens_padded[i, :texts_lens_list[i]] = torch.tensor(text_tokens)

        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate

        texts_tokens_padded = texts_tokens_padded.to(device)
        
        ori_text_cls_emb = self.clip_model_ori.ori_encode_text(texts_tokens_padded)

        return ori_text_cls_emb, texts_lens_list
    
    def bert_encode_text(self, raw_text):
        # enc_text = self.clip_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        enc_text, mask = self.clip_model(raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def cmdm_forward(self, x, timesteps, y=None):
        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.c_embed_timestep(timesteps)  # [1, bs, d]
        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text, texts_len_list = self.encode_text(y['text'])  ## [bs, 4x7, 512]
                if z_config.get_diy_config().training.use_gt_MB_simplified_data:
                    enc_text[:,1:,:] = y['motion_token_emb']
            text_emb=self.c_embed_text(self.mask_cond(enc_text, force_mask=force_mask))
            text_cls_emb = text_emb[:,:1,:]
            motion_token_emb=text_emb[:,1:,:]
            if self.emb_policy == 'add':
                emb = text_cls_emb.permute(1,0,2).contiguous() + time_emb

            x=self.c_input_process(x)
            ## 这里得处理motion_token_emb来匹配x的维度
            motion_token_emb=motion_token_emb.reshape(motion_token_emb.shape[0], 4, 7, -1).transpose(-2, -1)  # [64, 4, 512, 7]
            # self.c_joint_integration_layer: Linear(7 -> 1)
            motion_token_emb = self.c_joint_integration_layer(motion_token_emb).squeeze(-1)  # [64, 4, 512, 1]
            motion_token_emb = motion_token_emb.repeat_interleave(repeats=49, dim=1)  # [64, 4*49, 512]

            x += motion_token_emb.transpose(0,1)

            frames_mask = None
            is_valid_mask = y['mask'].shape[-1] > 1  # Don't use mask with the generate script
            if self.mask_frames and is_valid_mask:
                frames_mask = torch.logical_not(y['mask'][..., :x.shape[0]].squeeze(1).squeeze(1)).to(device=x.device)  ## 这里看着其实就是一个普通的取反操作
                if getattr(self, 'split_info_mode', False):
                    step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device) ## 这里是因为encoder中会在前面加上一维度，所以连带着mask也需要加上
                    frames_mask_enc = torch.cat([step_mask, frames_mask], dim=1)
                    frames_mask_dec = frames_mask
                elif self.emb_trans_dec or self.arch == 'trans_enc':
                    step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device) ## 这里是因为encoder中会在前面加上一维度，所以连带着mask也需要加上
                    frames_mask = torch.cat([step_mask, frames_mask], dim=1)

            if self.arch == 'trans_enc':
                # adding the timestep emb
                xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
                xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                control = []

                for i, layer in enumerate(self.c_seqTransEncoder.layers):
                    xseq = layer(xseq, src_key_padding_mask=frames_mask)
                    control.append(self.zero_convs[i](xseq))

        return control
    
    def _compute_debug_losses(self, lora_enc_cls, ori_CLIP_cls_emb, lora_CLIP_MB_rep, gt_MB_rep):
        """
        计算调试用的特征对齐 loss（不参与梯度）
        返回 dict: {'cls_lora_vs_clip': float, 'mb_lora_vs_gt': float}
        """
        debug_losses = {}
        with torch.no_grad():
            # 1. CLS token: LoRA vs Original CLIP
            if lora_enc_cls.shape == ori_CLIP_cls_emb.shape:
                cls_loss = torch.nn.functional.mse_loss(lora_enc_cls, ori_CLIP_cls_emb)
                debug_losses['cls_lora_vs_clip'] = cls_loss.item()
            else:
                print(f"[WARNING] CLS Shape mismatch: lora {lora_enc_cls.shape} vs ori {ori_CLIP_cls_emb.shape}")
                debug_losses['cls_lora_vs_clip'] = float('nan')

            # 2. MB rep: LoRA vs GT
            if lora_CLIP_MB_rep.shape == gt_MB_rep.shape:
                mb_loss = torch.nn.functional.mse_loss(lora_CLIP_MB_rep, gt_MB_rep)
                debug_losses['mb_lora_vs_gt'] = mb_loss.item()
            else:
                print(f"[WARNING] MB Shape mismatch: lora {lora_CLIP_MB_rep.shape} vs gt {gt_MB_rep.shape}")
                debug_losses['mb_lora_vs_gt'] = float('nan')

        return debug_losses

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        if z_config.get_diy_config().model.use_contronet_injection==True:
            assert self.arch=="trans_enc","ERROR: When using controlnet, MDM must be encoder"
            control = self.cmdm_forward(x, timesteps, y)

        bs, njoints, nfeats, nframes = x.shape
        time_emb = self.embed_timestep(timesteps)  # [1, bs, d]

        if 'target_cond' in y.keys():
            # NOTE: We don't use CFG for joints - but we do wat to support uncond sampling for generation and eval!
            time_emb += self.mask_cond(self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None], force_mask=y.get('target_uncond', False))  # For uncond support and CFG
            # time_emb += self.embed_target_cond(y['target_cond'], y['target_joint_names'], y['is_heading'])[None]  

        # Build input for prefix completion
        if self.is_prefix_comp:
            x = torch.cat([y['prefix'], x], dim=-1)
            y['mask'] = torch.cat([torch.ones([bs, 1, 1, self.context_len], dtype=y['mask'].dtype, device=y['mask'].device), 
                                   y['mask']], dim=-1)

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                lora_enc_cls_MB, texts_len_list = self.lora_clip_encode_text(y['text'])  ## [bs, 4x7, 512],如果使用cls_token第二维就是29
                lora_enc_cls = lora_enc_cls_MB[:,0:1,:]
                lora_CLIP_MB_rep = lora_enc_cls_MB[:,1:,:]
                ori_CLIP_cls_emb, _ = self.ori_clip_encode_text(y['text'])  ## 这个是普通CLIP出来的结果
                ## 想要在这里写一个保存loss的脚本
                
                if z_config.get_diy_config().training.use_gt_MB_simplified_data:
                    gt_MB_rep = y['motion_token_emb']
                    ori_CLIP_cls_emb, _ = self.ori_clip_encode_text(y['text'])
                    enc_text = torch.cat((ori_CLIP_cls_emb, gt_MB_rep), dim=1)

                if z_config.get_diy_config().training_input.use_end2end_ding_training:
                    if y.get("eval_time", False):
                        # eval 时，全 batch 都走 LoRA 分支
                        gt_MB_rep = y['motion_token_emb']
                        self.last_debug_losses = self._compute_debug_losses(
                                    lora_enc_cls, ori_CLIP_cls_emb, lora_CLIP_MB_rep, gt_MB_rep
                                )
                        if z_config.get_diy_config().training_input.use_cls_token:
                            lora_enc_cls = self.lra_clip_proj(lora_enc_cls)
                            if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                                lora_CLIP_MB_rep = self.lora_clip_MB_rep_proj(lora_CLIP_MB_rep)
                            enc_text = torch.cat((lora_enc_cls, lora_CLIP_MB_rep), dim=1)

                        elif z_config.get_diy_config().training_input.not_use_cls_token:
                            if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                                lora_CLIP_MB_rep = self.lora_clip_MB_rep_proj(lora_CLIP_MB_rep)
                            enc_text = lora_CLIP_MB_rep

                        elif z_config.get_diy_config().training_input.use_mean_MB_as_cls_in_gt_branch:
                            lora_enc_cls = self.lra_clip_proj(lora_enc_cls)
                            if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                                lora_CLIP_MB_rep = self.lora_clip_MB_rep_proj(lora_CLIP_MB_rep)
                            enc_text = torch.cat((lora_enc_cls, lora_CLIP_MB_rep), dim=1)

                        elif z_config.get_diy_config().training_input.both_use_ori_clip_cls_token:
                            ori_CLIP_cls_emb = self.ori_clip_proj(ori_CLIP_cls_emb)
                            if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                                lora_CLIP_MB_rep = self.lora_clip_MB_rep_proj(lora_CLIP_MB_rep)
                            enc_text = torch.cat((ori_CLIP_cls_emb, lora_CLIP_MB_rep), dim=1)

                    ## 端到端各种条件控制的实验
                    else:
                            gt_MB_rep = y['motion_token_emb']
                            self.last_debug_losses = self._compute_debug_losses(
                                    lora_enc_cls, ori_CLIP_cls_emb, lora_CLIP_MB_rep, gt_MB_rep
                                )
                            if z_config.get_diy_config().training_input.use_cls_token:
                                ori_CLIP_cls_emb = self.ori_clip_proj(ori_CLIP_cls_emb)
                                lora_enc_cls = self.lra_clip_proj(lora_enc_cls)
                                if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                                    lora_CLIP_MB_rep = self.lora_clip_MB_rep_proj(lora_CLIP_MB_rep)
                                    gt_MB_rep = self.MB_rep_proj(gt_MB_rep)
                                enc_text_sub_gt = torch.cat((ori_CLIP_cls_emb[:int(bs/2)], gt_MB_rep[:int(bs/2)]), axis=1)
                                enc_text_sub_lora = torch.cat((lora_enc_cls[:int(bs/2)], lora_CLIP_MB_rep[:int(bs/2)]), axis=1)
                                enc_text = torch.cat((enc_text_sub_gt, enc_text_sub_lora), dim=0)
                            
                            if z_config.get_diy_config().training_input.not_use_cls_token:
                                if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                                    lora_CLIP_MB_rep = self.lora_clip_MB_rep_proj(lora_CLIP_MB_rep)
                                    gt_MB_rep = self.MB_rep_proj(gt_MB_rep)
                                enc_text_sub_gt = gt_MB_rep[:int(bs/2)]
                                enc_text_sub_lora = lora_CLIP_MB_rep[:int(bs/2)]
                                enc_text = torch.cat((enc_text_sub_gt, enc_text_sub_lora), dim=0)
                            
                            if z_config.get_diy_config().training_input.use_mean_MB_as_cls_in_gt_branch:
                                ori_CLIP_cls_emb = torch.mean(gt_MB_rep, dim=1).unsqueeze(1)
                                lora_enc_cls = self.lra_clip_proj(lora_enc_cls)
                                if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                                    lora_CLIP_MB_rep = self.lora_clip_MB_rep_proj(lora_CLIP_MB_rep)
                                    gt_MB_rep = self.MB_rep_proj(gt_MB_rep)
                                enc_text_sub_gt = torch.cat((ori_CLIP_cls_emb[:int(bs/2)], gt_MB_rep[:int(bs/2)]), axis=1)
                                enc_text_sub_lora = torch.cat((lora_enc_cls[:int(bs/2)], lora_CLIP_MB_rep[:int(bs/2)]), axis=1)
                                enc_text = torch.cat((enc_text_sub_gt, enc_text_sub_lora), dim=0)
                            
                            if z_config.get_diy_config().training_input.both_use_ori_clip_cls_token:
                                ori_CLIP_cls_emb = self.ori_clip_proj(ori_CLIP_cls_emb)
                                if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
                                    lora_CLIP_MB_rep = self.lora_clip_MB_rep_proj(lora_CLIP_MB_rep)
                                    gt_MB_rep = self.MB_rep_proj(gt_MB_rep)
                                enc_text_sub_gt = torch.cat((ori_CLIP_cls_emb[:int(bs/2)], gt_MB_rep[:int(bs/2)]), axis=1)
                                enc_text_sub_lora = torch.cat((ori_CLIP_cls_emb[:int(bs/2)], lora_CLIP_MB_rep[:int(bs/2)]), axis=1)
                                enc_text = torch.cat((enc_text_sub_gt, enc_text_sub_lora), dim=0)

                
                ## 是否进一步pooling的对比实验
                if any([z_config.get_diy_config().data.Temperal_further_pooling, z_config.get_diy_config().data.Joint_further_pooling]):
                    if z_config.get_diy_config().data.Temperal_further_pooling and z_config.get_diy_config().data.Joint_further_pooling:
                        motion_token_emb=enc_text[:, -28:, :].reshape(enc_text.shape[0], 4, 7, -1)
                        motion_token_emb = motion_token_emb.mean(dim=1)  # [B, 7, D] -> 池化时间维度 (4->1)
                        motion_token_emb = motion_token_emb.mean(dim=1).unsqueeze(1)  # [B, D] -> 池化关节点维度 (7->1)
                        # 此时 shape: [B, 1, D]
                    elif z_config.get_diy_config().data.Temperal_further_pooling and (not z_config.get_diy_config().data.Joint_further_pooling):
                        motion_token_emb=enc_text[:, -28:, :].reshape(enc_text.shape[0], 4, 7, -1)
                        motion_token_emb = motion_token_emb.mean(dim=1)  # [B, 7, D] -> 时间维度池化 (4->1)
                        # 此时 shape: [B, 7, D]
                    elif (not z_config.get_diy_config().data.Temperal_further_pooling) and z_config.get_diy_config().data.Joint_further_pooling:
                        motion_token_emb=enc_text[:, -28:, :].reshape(enc_text.shape[0], 4, 7, -1)
                        motion_token_emb = motion_token_emb.mean(dim=2)  # [B, 4, D] -> 关节点维度池化 (7->1)
                        # 此时 shape: [B, 4, D]
                    enc_text = torch.cat((enc_text[:,:1,:], motion_token_emb), dim=1)
            if type(enc_text) == tuple: ## 不会跑
                enc_text, text_mask = enc_text
                if text_mask.shape[0] == 1 and bs > 1:  # casting mask for the single-prompt-for-all case
                    text_mask = torch.repeat_interleave(text_mask, bs, dim=0)
            ## 这里是有一个text的映射！！！得注意一下
            text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))  # casting mask for the single-prompt-for-all case
            if self.emb_policy == 'add':
                emb = text_emb.permute(1,0,2).contiguous() + time_emb
            else:
                emb = torch.cat([time_emb, text_emb], dim=0)
                text_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)
        if 'action' in self.cond_mode:  ## 不会跑
            action_emb = self.embed_action(y['action'])
            emb = time_emb + self.mask_cond(action_emb, force_mask=force_mask)
        if self.cond_mode == 'no_cond':    ## 不会跑
            # unconstrained
            emb = time_emb

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]
            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        # TODO - move to collate
        frames_mask = None
        is_valid_mask = y['mask'].shape[-1] > 1  # Don't use mask with the generate script
        if self.mask_frames and is_valid_mask:
            frames_mask = torch.logical_not(y['mask'][..., :x.shape[0]].squeeze(1).squeeze(1)).to(device=x.device)  ## 这里看着其实就是一个普通的取反操作
            if getattr(self, 'split_info_mode', False):
                step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device) ## 这里是因为encoder中会在前面加上一维度，所以连带着mask也需要加上
                frames_mask_enc = torch.cat([step_mask, frames_mask], dim=1)
                frames_mask_dec = frames_mask
            ## 我看起来这里其实主要是针对tran_enc
            elif self.emb_trans_dec or self.arch == 'trans_enc':
                if z_config.get_diy_config().model.use_contronet_injection:
                    step_mask = torch.zeros((bs, 1), dtype=torch.bool, device=x.device) ## 这里是因为encoder中会在前面加上一维度，所以连带着mask也需要加上
                    frames_mask = torch.cat([step_mask, frames_mask], dim=1)
                else:
                    if z_config.get_diy_config().model.clip_use_sen_emb:
                        step_mask = torch.zeros((bs, 29), dtype=torch.bool, device=x.device) ## 这里是因为encoder中会在前面加上一维度，所以连带着mask也需要加上
                        frames_mask = torch.cat([step_mask, frames_mask], dim=1)
                    else:
                        step_mask = torch.zeros((bs, 28), dtype=torch.bool, device=x.device) ## 这里是因为encoder中会在前面加上一维度，所以连带着mask也需要加上
                        frames_mask = torch.cat([step_mask, frames_mask], dim=1)

        if getattr(self, 'split_info_mode', False):
            cls_emb=emb[:1]
            motion_emb=emb[1:]
            xseq = torch.cat((cls_emb, x), axis=0)  # [seqlen+1, bs, d], emb之中是包含了文本和去噪步数的信息
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            cls_emb=xseq[:1]    ## 这个是文本的
            m_seq=xseq[1:]   ## 这个是那196维的motion seq
            
            version = self.split_version

            if version == 0:
                output = torch.cat((cls_emb, m_seq), axis=0)
                for layer in self.encoder_layers:
                    output = layer(output, src_key_padding_mask=frames_mask_enc)
                tgt = output[1:]  # 你可以根据需求修改这里的 tgt
                output = tgt
                for layer in self.decoder_layers:
                    output = layer(output, memory=motion_emb, tgt_key_padding_mask=frames_mask_dec)
            
            elif version == 1:
                output = m_seq
                for layer in self.decoder_layers:
                    output = layer(output, memory=motion_emb, tgt_key_padding_mask=frames_mask_dec)

                tgt = torch.cat((cls_emb, output), axis=0)
                for layer in self.encoder_layers:
                    tgt = layer(tgt, src_key_padding_mask=frames_mask_enc)
                output = tgt[1:]

            elif version == 2:
                # 交叉执行：一层 encoder，一层 decoder，一层 encoder，...
                encoder_layers = list(self.encoder_layers)
                decoder_layers = list(self.decoder_layers)
                layer_iter = []

                # 交替添加 encoder 和 decoder 层的迭代器
                for e_layer, d_layer in zip(encoder_layers, decoder_layers):
                    layer_iter.append(e_layer)
                    layer_iter.append(d_layer)
                
                output=m_seq
                output = torch.cat((cls_emb, output), axis=0)

                for layer in layer_iter:
                    if isinstance(layer, nn.TransformerEncoderLayer):
                        output = layer(output, src_key_padding_mask=frames_mask_enc)
                    elif isinstance(layer, nn.TransformerDecoderLayer):
                        output = layer(output, memory=motion_emb, tgt_key_padding_mask=frames_mask_enc) ## 这里没办法只能直接借用enc的mask了
                    else:
                        raise ValueError(f"Unexpected layer type: {type(layer)}")
                output = output[1:]

        else:
            if self.arch == 'trans_enc':
                # adding the timestep emb
                if z_config.get_diy_config().model.use_contronet_injection:
                    xseq = torch.cat((emb[:1], x), axis=0)
                    xseq = self.sequence_pos_encoder(xseq)
                    for i, layer in enumerate(self.seqTransEncoder.layers):
                        xseq=layer(xseq, src_key_padding_mask=frames_mask)
                        xseq=xseq+control[i]
                    output=xseq[1:]
                else:
                    xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
                    xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
                    output = self.seqTransEncoder(xseq, src_key_padding_mask=frames_mask)[-196:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

            elif self.arch == 'trans_dec':
                if self.emb_trans_dec:
                    xseq = torch.cat((time_emb, x), axis=0)
                else:
                    xseq = x
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

                if self.text_encoder_type == 'clip':
                    output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_key_padding_mask=frames_mask)
                elif self.text_encoder_type == 'bert':
                    output = self.seqTransDecoder(tgt=xseq, memory=emb, memory_key_padding_mask=text_mask, tgt_key_padding_mask=frames_mask)  # Rotem's bug fix
                else:
                    raise ValueError()

                if self.emb_trans_dec:
                    output = output[1:] # [seqlen, bs, d]

            elif self.arch == 'gru':
                xseq = x
                xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
                output, _ = self.gru(xseq)

        # Extract completed suffix
        if self.is_prefix_comp: ## 这边又不走，就不用管了
            output = output[self.context_len:]
            y['mask'] = y['mask'][..., self.context_len:]
        
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        if z_config.get_diy_config().training_input.use_end2end_ding_training:
             return output, self.last_debug_losses
        else:
            return output


    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output
    
class EmbedTargetLocSingle(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = len(self.extended_goal_joint_names) * 4  # 4 => (x,y,z,is_valid)
        self.latent_dim = latent_dim
        _layers = [nn.Linear(self.target_cond_dim, self.latent_dim)]
        for _ in range(num_layers):
            _layers += [nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)]
        self.mlp = nn.Sequential(*_layers)

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1).view(input.shape[0], -1)
        return self.mlp(mlp_input)


class EmbedTargetLocSplit(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim, num_layers=1):
        super().__init__()
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.target_cond_dim = 4
        self.latent_dim = latent_dim
        self.splited_dim = self.latent_dim // len(self.extended_goal_joint_names)
        assert self.latent_dim % len(self.extended_goal_joint_names) == 0
        self.mini_mlps = nn.ModuleList()
        for _ in self.extended_goal_joint_names:
            _layers = [nn.Linear(self.target_cond_dim, self.splited_dim)]
            for _ in range(num_layers):
                _layers += [nn.SiLU(), nn.Linear(self.splited_dim, self.splited_dim)]
            self.mini_mlps.append(nn.Sequential(*_layers))

    def forward(self, input, target_joint_names, target_heading):
        # TODO - generate validity from outside the model
        validity = torch.zeros_like(input)[..., :1]
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            for j in sample_joint_names_w_heading:
                validity[sample_idx, self.extended_goal_joint_names.index(j)] = 1.

        mlp_input = torch.cat([input, validity], dim=-1)
        mlp_splits = [self.mini_mlps[i](mlp_input[:, i]) for i in range(mlp_input.shape[1])] 
        return torch.cat(mlp_splits, dim=-1)
  
class EmbedTargetLocMulti(nn.Module):
    def __init__(self, all_goal_joint_names, latent_dim):
        super().__init__()
        
        # todo: use a tensor of weight per joint, and another one for biases, then apply a selection in one go like we to for actions
        self.extended_goal_joint_names = all_goal_joint_names + ['traj', 'heading']
        self.extended_goal_joint_idx = {joint_name: idx for idx, joint_name in enumerate(self.extended_goal_joint_names)}
        self.n_extended_goal_joints = len(self.extended_goal_joint_names)
        self.target_loc_emb = nn.ParameterDict({joint_name: 
            nn.Sequential(
                nn.Linear(3, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim)) 
            for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
            # nn.Linear(3, latent_dim) for joint_name in self.extended_goal_joint_names})  # todo: check if 3 works for heading and traj
        self.target_all_loc_emb = WeightedSum(self.n_extended_goal_joints) # nn.Linear(self.n_extended_goal_joints, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, input, target_joint_names, target_heading):
        output = torch.zeros((input.shape[0], self.latent_dim), dtype=input.dtype, device=input.device)
        
        # Iterate over the batch and apply the appropriate filter for each joint
        for sample_idx, sample_joint_names in enumerate(target_joint_names):
            sample_joint_names_w_heading = np.append(sample_joint_names, 'heading') if target_heading[sample_idx] else sample_joint_names
            output_one_sample = torch.zeros((self.n_extended_goal_joints, self.latent_dim), dtype=input.dtype, device=input.device)
            for joint_name in sample_joint_names_w_heading:
                layer = self.target_loc_emb[joint_name]
                output_one_sample[self.extended_goal_joint_idx[joint_name]] = layer(input[sample_idx, self.extended_goal_joint_idx[joint_name]])  
            output[sample_idx] = self.target_all_loc_emb(output_one_sample)
            # print(torch.where(output_one_sample.sum(axis=1)!=0)[0].cpu().numpy())
               
        return output
