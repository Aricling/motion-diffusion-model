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

def clip_finetune_l2_loss(model_output, raw_text_output, MB_targets, lens_list):
    batch_size, seq_len, dim = model_output.shape
    lens_tensor = torch.tensor(lens_list, device=model_output.device)
    
    # Mask generation for clip loss computation
    mask = torch.arange(seq_len, device=model_output.device).expand(batch_size, seq_len) < lens_tensor.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()
    
    # Clip loss computation
    diff_clip = model_output - raw_text_output
    loss_clip = ((diff_clip ** 2) * mask).sum() / (mask.sum() * dim)
    
    # Motion token loss computation using batch processing
    # Construct a tensor to hold all `lens_sub + 1` and `lens_sub + 1 + 28` indices
    valid_motion_indices = torch.stack([
        lens_tensor + 1,
        lens_tensor + 1 + 28
    ], dim=1)  # Shape: [batch_size, 2]

    # Use indices to gather the valid motion token slices from model_output
    # Note: valid_motion_indices[:, 0] means start index, valid_motion_indices[:, 1] means end index
    motion_slices = torch.stack([
        model_output[b_idx, valid_motion_indices[b_idx, 0]:valid_motion_indices[b_idx, 1]]
        for b_idx in range(batch_size)
    ], dim=0)  # Shape: [batch_size, 28, dim]

    diff_motion = motion_slices - MB_targets  # Compute differences, [batch_size, 28, dim]
    
    loss_motion_token = (diff_motion ** 2).sum() / diff_motion.numel()
    
    loss_dict = {
        'loss_clip': loss_clip,
        'loss_motion_token': loss_motion_token
    }
    
    return loss_dict
