from diffusion.nn import mean_flat, sum_flat
import torch
import numpy as np

def angle_l2(angle1, angle2):
    a = angle1 - angle2
    a = (a + (torch.pi/2)) % torch.pi - (torch.pi/2)
    return a ** 2

def diff_l2(a, b):
    return (a - b) ** 2

def masked_l2(a, b, mask, loss_fn=diff_l2, epsilon=1e-8, entries_norm=True):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    loss = loss_fn(a, b)
    loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1]
    if len(a.shape) > 3:
        n_entries *= a.shape[2]
    non_zero_elements = sum_flat(mask)
    if entries_norm:
        # In cases the mask is per frame, and not specifying the number of entries per frame, this normalization is needed,
        # Otherwise set it to False
        non_zero_elements *= n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / (non_zero_elements + epsilon)  # Add epsilon to avoid division by zero
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val


def masked_goal_l2(pred_goal, ref_goal, cond, all_goal_joint_names):
    all_goal_joint_names_w_traj = np.append(all_goal_joint_names, 'traj')
    target_joint_idx = [[np.where(all_goal_joint_names_w_traj == j)[0][0] for j in sample_joints] for sample_joints in cond['target_joint_names']]
    loc_mask = torch.zeros_like(pred_goal[:,:-1], dtype=torch.bool)
    for sample_idx in range(loc_mask.shape[0]):
        loc_mask[sample_idx, target_joint_idx[sample_idx]] = True
    loc_mask[:, -1, 1] = False  # vertical joint of 'traj' is always masked out
    loc_loss = masked_l2(pred_goal[:,:-1], ref_goal[:,:-1], loc_mask, entries_norm=False)
    
    heading_loss = masked_l2(pred_goal[:,-1:, :1], ref_goal[:,-1:, :1], cond['is_heading'].unsqueeze(1).unsqueeze(1), loss_fn=angle_l2, entries_norm=False)

    loss =  loc_loss + heading_loss
    return loss

def clip_finetune_l2_loss(model_output, x_motion_proj, raw_text_output, MB_targets, texts_lens_list, motion_lens_list):
    batch_size, seq_len, dim = model_output.shape
    texts_lens_tensor = torch.tensor(texts_lens_list, device=model_output.device)   ## 这里指的是text的length
    motion_lens_tensor = torch.tensor(motion_lens_list, device=model_output.device) # motion frame lengths

    # ----- Clip Loss -----
    # Mask generation for clip loss computation
    mask = torch.arange(seq_len, device=model_output.device).expand(batch_size, seq_len) < texts_lens_tensor.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()
    
    # Clip loss computation
    diff_clip = model_output - raw_text_output
    loss_clip = ((diff_clip ** 2) * mask).sum() / (mask.sum() * dim)
    
    # ----- Motion Token Loss -----
    diff_motion = x_motion_proj - MB_targets  # Compute differences, [batch_size, 49, 7, 32]
    
    # ----- Motion Mask Generation -----
    frame_group_count = 49
    frames_per_group = 4
    tokens_per_group = 7
    B, groups, num_joints, out_dim = x_motion_proj.shape    ## [batch_size, 49, 7, 32]

    # ----- 生成 motion 有效 mask -----
    # [1, 49]，表示每个 group 对应的帧数阈值
    fuse_mask = torch.arange(1, frame_group_count + 1, device=model_output.device).view(1, -1) * frames_per_group  

    # motion_lens_tensor: [B] → [B, 1]，和 [1, 49] 广播比较
    fuse_mask = motion_lens_tensor.view(-1, 1) >= fuse_mask    # [B, 49]，bool

    # 展开到和 diff_motion 一样的 shape: [B, 49, 7, 32]
    motion_mask = fuse_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_joints, out_dim).float()

    # ----- Motion Loss -----
    denom = motion_mask.sum()
    if denom > 0:
        loss_motion_token = ((diff_motion ** 2) * motion_mask).sum() / denom
    else:
        loss_motion_token = torch.tensor(0., device=diff_motion.device)

    loss_dict = {
        'loss_clip': loss_clip,
        'loss_motion_token': loss_motion_token
    }

    return loss_dict
