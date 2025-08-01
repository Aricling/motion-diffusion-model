import torch
import torch.nn as nn

class STPool(nn.Module):
    """
    Skeleto-Temporal Pooling.
    """
    def __init__(
        self,
        dataset="t2m",
        temporal_stride = None
    ):
        if not dataset in ["t2m", "kit"]:
            raise ValueError("dataset should be 't2m' or 'kit'")
        
        super(STPool, self).__init__()

        self.skeleton_pool_0, self.skeleton_mapping_0 = self._get_skeleton_pooling(dataset, depth=0)
        self.skeleton_pool_0 = nn.Parameter(self.skeleton_pool_0, requires_grad=False) # [J_out, J_in]

        self.skeleton_pool_1, self.skeleton_mapping_1 = self._get_skeleton_pooling(dataset, depth=1)
        self.skeleton_pool_1 = nn.Parameter(self.skeleton_pool_1, requires_grad=False) # [J_out, J_in]
        
        self.temporal_pool = nn.AvgPool1d(kernel_size=temporal_stride, stride=temporal_stride)
    
    def _get_skeleton_pooling(self, dataset='t2m', depth=0):
        if depth == 0:
            if dataset == "t2m":
                weight = torch.zeros(12, 17)
                # mapping = [
                #     [(0, 1, 2, 3), 0],       # root
                #     [(0, 1, 4), 1],          # left hip
                #     [(4, 7, 10), 2],         # left leg
                #     [(0, 2, 5), 3],          # right hip
                #     [(5, 8, 11), 4],         # right leg
                #     [(0, 3, 6, 9), 5],       # spine
                #     [(9, 6, 12, 13, 14), 6], # chest
                #     [(9, 13, 16), 7],        # left shoulder
                #     [(16, 18, 20), 8],       # left arm
                #     [(9, 14, 17), 9],        # right shoulder
                #     [(17, 19, 21), 10],      # right arm
                #     [(9, 12, 15), 11],       # head
                # ]
                mapping = [
                    [(0, 4, 1, 7), 0],
                    [(0, 4, 5), 1],
                    [(5, 6), 2],
                    [(0, 1, 2), 3],
                    [(2, 3), 4],
                    [(0, 7, 8), 5],
                    [(8, 9), 6],
                    [(8, 11), 7],
                    [(11, 12, 13), 8],
                    [(8, 14), 9],
                    [(14, 15, 16), 10],
                    [(8, 9, 10), 11],
                ]
            else:
                weight = torch.zeros(12, 21)
                mapping = [
                    [(0, 1, 11, 16), 0],   # root
                    [(0, 16, 17, 18), 1],  # left hip
                    [(17, 18, 19, 20), 2], # left leg
                    [(0, 11, 12, 13), 3],  # right hip
                    [(12, 13, 14, 15), 4], # right leg
                    [(0, 1, 2, 3), 5],     # spine
                    [(2, 3, 5, 8, 4), 6],  # chest
                    [(3, 8, 9), 7],        # left shoulder
                    [(8, 9, 10), 8],       # left arm
                    [(3, 5, 6), 9],        # right shoulder
                    [(5, 6, 7), 10],       # right arm
                    [(3, 4), 11],          # head
                ]
        elif depth==1:
            weight = torch.zeros(7, 12)
            mapping = [
                [(0, 1, 3, 5), 0], # root
                [(0, 1, 2),    1], # left lower
                [(0, 3, 4),    2], # right lower
                [(0, 5, 6),    3], # spine
                [(6, 7, 8),    4], # left upper
                [(6, 9, 10),   5], # right upper
                [(6, 11),      6], # head
            ]
        
        for joints, idx in mapping:
            weight[idx, joints] = 1
        weight = weight / weight.sum(dim=1, keepdim=True)

        return weight, mapping

    def forward(self, x):
        """
        x: [B, T, J, D]
        out: [B, T // 2, J_out, D]
        """
        B, T, J_in, D = x.size()

        # skeleton pooling
        out_0 = torch.matmul(self.skeleton_pool_0, x) # [B, T, J_out, D]
        out_1 = torch.matmul(self.skeleton_pool_1, out_0) # [B, T, J_out, D]
        J_out = out_1.size(2)

        # temporal pooling
        out = out_1.permute(0, 2, 3, 1).reshape(B * J_out, D, T)
        out = self.temporal_pool(out)
        out = out.reshape(B, J_out, D, -1).permute(0, 3, 1, 2) # [B, T // 2, J_out, D]

        return out