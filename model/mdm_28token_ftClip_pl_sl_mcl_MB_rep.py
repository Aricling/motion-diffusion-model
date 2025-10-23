import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from model.rotation2xyz import Rotation2xyz
from model.BERT.BERT_encoder_lora import load_bert_lora
from model.BERT.BERT_encoder import load_bert

from utils.misc import WeightedSum
import z_config
from clip.utils.add_lora import prepare_lora_clip_model
import contextlib

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
                    # self.clip_model = load_bert_lora(bert_model_path)  # Sorry for that, the naming is for backward compatibility
                    self.clip_model_ori = load_bert(bert_model_path)
                    self.clip_dim = 768
                else:
                    raise ValueError('We only support [CLIP, BERT] text encoders') 
                self.diy_latent_dim = int(z_config.get_diy_config().ori_CLIP_cls_emb_proj_test_1012.target_dim)
                
                # self.embed_text = nn.Linear(self.diy_latent_dim, self.latent_dim) ## 对于cls token的投影

                # self.motion_token_dim_proj1 = nn.Linear(self.diy_latent_dim, 32)    ## motion token送入MDM前的投影
                # self.motion_token_dim_proj2 = nn.Linear(self.latent_dim, self.latent_dim)

                self.ori_bert_out_emb_proj_layer = nn.Linear(self.clip_dim, self.diy_latent_dim)    ## 对Bert出来结果的投影

                self.gt_3d_motion_emb_proj = nn.Linear(32, self.latent_dim) ## 对于VAE gt motion的投影

                self.pred_for_gt_loss_proj = nn.Linear(self.diy_latent_dim, 32)
                
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

        # if z_config.get_diy_config().training_input.use_end2end_ding_training:
        #     ## 这里相当于第一个实验和第二个实验
        #     if z_config.get_diy_config().training_input.use_cls_token:
        #         self.ori_clip_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #         self.lra_clip_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #         if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
        #             self.MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #             self.lora_clip_MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #     if z_config.get_diy_config().training_input.not_use_cls_token:
        #         if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
        #             self.MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #             self.lora_clip_MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
            
        #     ## 这里是第三个实验
        #     if z_config.get_diy_config().training_input.use_mean_MB_as_cls_in_gt_branch:
        #         self.mean_motion_bert_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #         self.lra_clip_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #         if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
        #             self.MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #             self.lora_clip_MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)

        #     ## 这里是第四个实验
        #     if z_config.get_diy_config().training_input.both_use_ori_clip_cls_token:
        #         self.ori_clip_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #         if z_config.get_diy_config().training_input.add_proj_linear_to_MB_rep:
        #             self.MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
        #             self.lora_clip_MB_rep_proj = nn.Linear(self.clip_dim, self.latent_dim)
        # if int(z_config.get_diy_config().VAE_emb_shape_gt.pooled_dim) !=4:
        #     self.extended_proj_model = ExtendedTransformerDecoder(embed_dim=self.diy_latent_dim, num_heads=8, num_layers=3, add_tokens=7*int(z_config.get_diy_config().VAE_emb_shape_gt.pooled_dim))
        # else:
        #     self.extended_proj_model = ExtendedTransformerDecoder(embed_dim=self.diy_latent_dim, num_heads=8, num_layers=3, add_tokens=28)

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
    
    def _compute_debug_losses(
        self,
        lora_enc_cls,
        ori_CLIP_cls_emb,
        lora_CLIP_MB_rep,
        gt_MB_rep,
        motion_lens_list,
        **kwargs
    ):
        """
        计算调试用的特征对齐 loss。
        - 如果 cls_and_MB_extra_supervision_loss = True，返回可梯度回传的 tensor；
        - 否则返回 float 值（.item()），仅用于监控。
        """
        device = lora_enc_cls.device
        motion_lens_tensor = motion_lens_list.detach().clone().to(device)
        debug_losses = {}

        use_supervision_loss = z_config.get_diy_config().training_input.cls_and_MB_extra_supervision_loss
        ctx = torch.no_grad() if not use_supervision_loss else contextlib.nullcontext()

        def _finalize_loss(loss_value):
            """根据模式返回 tensor 或 float"""
            return loss_value if use_supervision_loss else loss_value.item()

        def _zero_loss_like(x):
            """返回一个 0 loss（支持梯度回传）"""
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        with ctx:
            # -------------------------------------------------
            # 1. CLS token: LoRA vs Original CLIP
            # -------------------------------------------------
            if lora_enc_cls.shape == ori_CLIP_cls_emb.shape:
                cls_loss = torch.nn.functional.mse_loss(lora_enc_cls, ori_CLIP_cls_emb)
                debug_losses["cls_lora_vs_clip"] = _finalize_loss(cls_loss)
            elif self.text_encoder_type == "bert":
                text_mask = kwargs.get('text_mask')
                squared_error = ((lora_enc_cls[:len(ori_CLIP_cls_emb)] - ori_CLIP_cls_emb) ** 2) # *(~text_mask.unsqueeze(2).permute(1,0,2))
                valid_mask = (~text_mask).unsqueeze(-1).permute(1,0,2)
                masked_squared_error = squared_error * valid_mask
                total_valid_elements = valid_mask.sum() * squared_error.size(-1)  # [T*B*mask==1] * D
                cls_loss = masked_squared_error.sum() / (total_valid_elements + 1e-8)
                debug_losses["cls_lora_vs_clip"] = _finalize_loss(cls_loss)

            else:
                print(f"[WARNING] CLS Shape mismatch: lora {lora_enc_cls.shape} vs ori {ori_CLIP_cls_emb.shape}")
                debug_losses["cls_lora_vs_clip"] = _zero_loss_like(lora_enc_cls) if use_supervision_loss else float("nan")

            # -------------------------------------------------
            # 2. MB rep: LoRA vs GT
            # -------------------------------------------------
            if lora_CLIP_MB_rep.shape == gt_MB_rep.shape:
                B, T, D = lora_CLIP_MB_rep.shape

                # 构造 motion mask [B, 28, D]
                frame_group_count, frames_per_group, tokens_per_group = 4, 49, 7
                fuse_mask_4 = (
                    motion_lens_tensor.view(-1, 1) >= torch.arange(1, frame_group_count + 1, device=device) * frames_per_group
                )  # [B, 4]
                motion_mask = (
                    fuse_mask_4.unsqueeze(-1)
                    .expand(-1, -1, tokens_per_group)
                    .reshape(B, 28)
                    .unsqueeze(-1)
                    .expand(B, 28, D)
                    .float()
                )

                # 逐元素 MSE 并应用 mask
                beta = float(z_config.get_diy_config().training_input.motion_centric_loss_strength)
                target = (1 - beta) * gt_MB_rep.detach() + beta * gt_MB_rep  # (1-β)不传梯度，β部分有梯度
                elementwise_loss = (lora_CLIP_MB_rep - target) ** 2
                masked_loss = elementwise_loss * motion_mask

                # 归一化
                num_valid_elements = motion_mask.sum()
                if num_valid_elements > 0:
                    mb_loss = masked_loss.sum() / num_valid_elements
                else:
                    mb_loss = _zero_loss_like(lora_CLIP_MB_rep)

                debug_losses["mb_lora_vs_gt"] = _finalize_loss(mb_loss)
            else:
                print(f"[WARNING] MB Shape mismatch: lora {lora_CLIP_MB_rep.shape} vs gt {gt_MB_rep.shape}")
                debug_losses["mb_lora_vs_gt"] = _zero_loss_like(lora_CLIP_MB_rep) if use_supervision_loss else float("nan")

        return debug_losses

    def extract_special_range(self, enc_text, false_counts, start_offset=-31, end_offset=-3):
        """
        提取从 false_counts+start_offset 到 false_counts+end_offset 区间的 token embedding

        参数:
            enc_text: Tensor, shape [seq_len, batch, hidden_dim] 或 [batch, seq_len, hidden_dim]
            false_counts: Tensor, shape [batch]，表示每个样本中 False 的数量
            start_offset: int，起始相对偏移 (包含)
            end_offset: int，结束相对偏移 (包含)

        返回:
            special_tokens: list[Tensor]，长度=batch，每个元素 [num_tokens, hidden_dim]
                            由于每个样本区间长度可能不一样，所以返回 list 而不是 tensor
        """
        # 转换成 [batch, seq_len, hidden_dim]
        if enc_text.shape[0] != false_counts.shape[0]:
            enc_text = enc_text.transpose(0, 1)

        batch_size, seq_len, hidden_dim = enc_text.shape
        device = enc_text.device

        special_tokens = []
        for b in range(batch_size):
            start_idx = false_counts[b].item() + start_offset
            end_idx   = false_counts[b].item() + end_offset
            if start_idx < 0 or end_idx >= seq_len:
                raise ValueError(f"batch {b}: 索引越界 ({start_idx}, {end_idx})")
            tokens = enc_text[b, start_idx:end_idx]  # [num_tokens, hidden_dim]
            special_tokens.append(tokens)
        MB_tokens = torch.stack(special_tokens)

        return MB_tokens

    def compute_mb_rep_loss(self, lora_CLIP_MB_rep, gt_MB_rep, motion_lens_tensor, align_strength=1.0):
        """
        计算 MB 表征的单向监督 loss（lora_CLIP_MB_rep 向 gt_MB_rep 对齐）
        Args:
            lora_CLIP_MB_rep (torch.Tensor): LoRA motion feature [B, T, D]
            gt_MB_rep (torch.Tensor): GT motion feature [B, T, D]
            motion_lens_tensor (torch.Tensor): motion长度 [B]
            align_strength (float): 对齐强度 β ∈ [0,1]，0 表示不监督，1 表示完全监督
        """
        device = lora_CLIP_MB_rep.device
        debug_losses = {}
        use_supervision_loss = z_config.get_diy_config().training_input.cls_and_MB_extra_supervision_loss

        def _finalize_loss(loss_value):
            return loss_value if use_supervision_loss else loss_value.item()

        def _zero_loss_like(x):
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # 形状检查
        if lora_CLIP_MB_rep.shape != gt_MB_rep.shape:
            print(f"[WARNING] MB Shape mismatch: lora {lora_CLIP_MB_rep.shape} vs gt {gt_MB_rep.shape}")
            debug_losses["mb_lora_vs_gt"] = (
                _zero_loss_like(lora_CLIP_MB_rep) if use_supervision_loss else float("nan")
            )
            return debug_losses

        B, T, D = lora_CLIP_MB_rep.shape

        # -------------------------------------------------
        # 构造 motion mask [B, 28, D]
        # -------------------------------------------------
        frame_group_count, frames_per_group, tokens_per_group = 4, 49, 7
        fuse_mask_4 = (
            motion_lens_tensor.view(-1, 1)
            >= torch.arange(1, frame_group_count + 1, device=device) * frames_per_group
        )  # [B, 4]
        motion_mask = (
            fuse_mask_4.unsqueeze(-1)
            .expand(-1, -1, tokens_per_group)
            .reshape(B, 28)
            .unsqueeze(-1)
            .expand(B, 28, D)
            .float()
        )

        # -------------------------------------------------
        # 单向 loss: lora_CLIP_MB_rep -> gt_MB_rep
        # L = || zt - (1 - β)*sg(zm) - β*zm ||^2
        # -------------------------------------------------
        beta = align_strength
        target = (1 - beta) * gt_MB_rep.detach() + beta * gt_MB_rep  # (1-β)不传梯度，β部分有梯度
        elementwise_loss = (lora_CLIP_MB_rep - target) ** 2
        masked_loss = elementwise_loss * motion_mask

        num_valid_elements = motion_mask.sum()
        if num_valid_elements > 0:
            mb_loss = masked_loss.sum() / num_valid_elements
        else:
            mb_loss = _zero_loss_like(lora_CLIP_MB_rep)

        debug_losses["mb_lora_vs_gt"] = _finalize_loss(mb_loss)
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
            # if 'text_embed' in y:  # caching option
            #     enc_text = y['text_embed']
            lora_enc_cls_MB, texts_len_list = self.lora_clip_encode_text(y['text'])
            lora_enc_cls = lora_enc_cls_MB[:, :1, :]
            lora_CLIP_MB_rep = lora_enc_cls_MB[:, 1:, :]
            ori_CLIP_cls_emb, _ = self.ori_clip_encode_text(y['text'])
            gt_MB_rep = y.get('motion_token_emb', None)
            if gt_MB_rep is not None:
                self.last_debug_losses = self._compute_debug_losses(
                    lora_enc_cls, ori_CLIP_cls_emb, lora_CLIP_MB_rep, gt_MB_rep, y['lengths']
                )
                
            if z_config.get_diy_config().training_input.use_cls_token:  # 当前使用 BERT 的 gt
                # pred_motion_tokens = self.motion_token_dim_proj1(pred_motion_tokens)

                if z_config.get_diy_config().training_input.use_cls_token:  # 当前使用 BERT 的 gt
                    eval_time = y.get("eval_time", False)
                    use_gt = z_config.get_diy_config().training_input.use_gt_for_training
                    use_MB_gt = True ## 是否使用MB的gt，False就为默认VAE的gt

                    if use_MB_gt:
                        gt_3d_rep = y['motion_token_emb'].reshape(-1, 4, 7, 512)
                    else:
                        gt_3d_rep = y["pooled_3d_emb_gt"]  # [64, 4, 7, 32] ## VAE的结果

                    pred_motion_tokens = lora_CLIP_MB_rep.permute(1, 0, 2)

                    bs = pred_motion_tokens.shape[1]
                    pred_branch = pred_motion_tokens  # [28, 64, 32]

                    # ---- 统一处理 gt branch ----
                    if gt_3d_rep is not None:
                        mask_mode = int(z_config.get_diy_config().training_input.gt_mask_mode)  # 1/2/3
                        mask_prob = float(z_config.get_diy_config().training_input.mask_probility)
                        bs, T, J, C = gt_3d_rep.shape  # e.g. bs,4,7,32
                        num_tokens = T * J  # should equal pred_branch.shape[0], i.e. 28

                        if not eval_time:
                            # ---------- 生成共享 mask (token 级) ----------
                            # 我们生成一个 token_mask_flat: shape (bs, T*J) -> 对应 gt 的 flatten token (t,j)
                            if mask_mode == 1:
                                # time mask: 只按时间维决定某个时间 step 是否全部置0
                                time_mask = (torch.rand(bs, T, device=gt_3d_rep.device) < mask_prob)  # (bs, T)
                                # expand 为 (bs, T, J) 再展平
                                token_mask = time_mask[:, :, None].expand(-1, -1, J)  # (bs, T, J)
                            elif mask_mode == 2:
                                # joint mask: 只按关节维决定某个关节是否全部置0
                                joint_mask = (torch.rand(bs, J, device=gt_3d_rep.device) < mask_prob)  # (bs, J)
                                token_mask = joint_mask[:, None, :].expand(-1, T, -1)  # (bs, T, J)
                            elif mask_mode == 3:
                                # token 级 mask: 每个 (t,j) 单独随机
                                token_mask = (torch.rand(bs, T, J, device=gt_3d_rep.device) < mask_prob)  # (bs, T, J)
                            else:
                                raise ValueError(f"Unknown mask_mode {mask_mode}")

                            # 统一转换为 (bs, T, J, 1) 以便直接作用到 gt_3d_rep
                            token_mask_gt = token_mask[:, :, :, None]  # (bs, T, J, 1)

                            # ---------- 应用到 gt ----------
                            gt_3d_rep = gt_3d_rep.masked_fill(token_mask_gt, 0.0)  # same mask

                            # ---------- 将 token_mask 展平成 (bs, T*J) 并应用到 pred ----------
                            token_mask_flat = token_mask.reshape(bs, -1)  # (bs, T*J)

                            # pred_branch: [T_pred, bs, C_pred] -> permute到 [bs, T_pred, C_pred]
                            pred_branch = pred_branch.permute(1, 0, 2)  # [bs, T*J, C_pred]
                            pred_branch = pred_branch.masked_fill(token_mask_flat[:, :, None], 0.0)
                            pred_branch = pred_branch.permute(1, 0, 2)  # [T*J, bs, C_pred]

                        # ---- reshape gt branch 并映射维度与原逻辑保持一致 ----
                        gt_3d_rep = gt_3d_rep.reshape(bs, -1, C)  # [bs, T*J, C]
                        # gt_3d_rep = self.motion_token_dim_proj2(gt_3d_rep)  # map channels
                        gt_branch = gt_3d_rep.permute(1, 0, 2)  # [T*J, bs, C_out]
                        # pred_branch = self.motion_token_dim_proj2(pred_branch)

                    # ---- train / eval 下 text_emb 的组合逻辑不变 ----
                    if not eval_time:
                        # if gt_3d_rep is not None:
                            # self.last_debug_losses = {}
                            # self.last_debug_losses["mb_lora_vs_gt"] = torch.tensor(0.0, device=gt_3d_rep.device)

                        if not z_config.get_diy_config().training_input.all_batch_using_lora_clip_out:
                            text_emb = gt_branch if use_gt else torch.cat(
                                (gt_branch[:, :bs // 2], pred_branch[:, bs // 2:]), dim=1
                            )
                        else:
                            text_emb = pred_branch
                    else:
                        if not z_config.get_diy_config().training_input.all_batch_using_lora_clip_out:
                            text_emb = gt_branch if use_gt else pred_branch
                        else:
                            text_emb = gt_branch if use_gt else pred_branch

                
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

            ## 这里是有一个text的映射！！！得注意一下
            text_emb = self.mask_cond(text_emb, force_mask=force_mask)  # casting mask for the single-prompt-for-all case

            if self.emb_policy == 'add':
                emb = text_emb + time_emb
            else:
                emb = torch.cat([time_emb, text_emb], dim=0)
                text_mask = torch.cat([torch.zeros_like(text_mask[:, 0:1]), text_mask], dim=1)
        if 'action' in self.cond_mode:  ## 不会跑
            action_emb = self.embed_action(y['action'])
            emb = time_emb + self.mask_cond(action_emb, force_mask=force_mask)
        if self.cond_mode == 'no_cond':    ## 不会跑
            # unconstrained
            emb = time_emb

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
                    # expanded_mask = self.expand_text_mask(text_mask_ori, total_len=78)
                    output = self.seqTransDecoder(tgt=xseq, memory=emb, memory_key_padding_mask=None, tgt_key_padding_mask=frames_mask)  # Rotem's bug fix
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
    
    def expand_text_mask(self, text_mask_ori, total_len=78):
        """
        将 text_mask_ori 从原长度扩展到 total_len，
        在末尾补充 False（表示有效，不屏蔽）。
        """
        B, L = text_mask_ori.shape
        if L >= total_len:
            # 截断（一般不会发生，但安全起见）
            return text_mask_ori[:, :total_len]

        pad_len = total_len - L
        pad_part = torch.zeros(B, pad_len, dtype=torch.bool, device=text_mask_ori.device)
        expanded_mask = torch.cat([text_mask_ori, pad_part], dim=1)
        return expanded_mask

    def add_motion_tokens_lora(self, texts, motion_token="<motion_token>", start_token="<start_of_motion>", end_token="<end_of_motion>", num_start=28):
        processed = []
        for t in texts:
            new_t = t + " " + start_token + " " + " ".join([motion_token] * num_start) + " " + end_token
            processed.append(new_t)
        return processed
        
    def bert_encode_text_lora(self, raw_text):
        # enc_text = self.clip_model(raw_text)
        # enc_text = enc_text.permute(1, 0, 2)
        # return enc_text
        texts_with_tokens = self.add_motion_tokens_lora(raw_text)
        enc_text, mask = self.clip_model(texts_with_tokens)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def bert_encode_text_ori(self, raw_text):
        enc_text, mask = self.clip_model_ori(raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask  # mask: True means no token there, we invert since the meaning of mask for transformer is inverted  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        return enc_text, mask

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)

class CLSTokenTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=1):
        super().__init__()
        self.embed_dim = embed_dim

        # 可学习的全局 token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.cls_token)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=False,  # 输入 (seq_len, batch, embed)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [seq_len=50, batch=64, embed=768]
        返回:
            global_token_output: [1, batch, embed] 经过 self-attn 后的第0号 token
        """
        seq_len, batch_size, _ = x.size()

        # 扩展 cls_token 到 batch_size
        cls_tok = self.cls_token.expand(-1, batch_size, -1)  # [1, 64, 768]

        # 拼接 cls_token + 原始序列
        combined = torch.cat([cls_tok, x], dim=0)  # [51, 64, 768]

        # 经过 Transformer Encoder
        output = self.transformer_encoder(combined)  # [51, 64, 768]

        # 取出第0号 token（cls_token 对应）
        global_token_output = output[0:1, :, :]  # [1, 64, 768]

        return global_token_output


class ExtendedTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=1, add_tokens=28):
        super().__init__()
        self.embed_dim = embed_dim
        self.add_tokens = add_tokens

        # learnable motion queries
        self.learnable_tokens = nn.Parameter(torch.randn(add_tokens, 1, embed_dim))
        nn.init.xavier_uniform_(self.learnable_tokens)

        # Decoder only (no encoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        x: memory from BERT (already contextualized), shape (S, B, D)
        """
        seq_len, batch_size, _ = x.size()

        # motion query tokens
        tgt = self.learnable_tokens.expand(-1, batch_size, -1)  # (T_add, B, D)

        # decoder attends to BERT features (memory)
        out = self.transformer_decoder(
            tgt, x,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        return out  # (T_add, B, D)


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
