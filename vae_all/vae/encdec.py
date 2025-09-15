import torch
import torch.nn as nn

from vae_all.skeleton.conv import ResSTConv, get_activation
from vae_all.skeleton.pool import STPool, STUnpool
from vae_all.utils.paramUtil import kit_adj_list, t2m_adj_list, h36m_adj_list
from vae_all.utils.skeleton import adj_list_to_edges


class MotionEncoder(nn.Module):
    def __init__(self, opt):
        super(MotionEncoder, self).__init__()

        self.pose_dim = opt.pose_dim
        # self.joints_num = (self.pose_dim + 1) // 12
        self.joints_num = 17
        self.latent_dim = opt.latent_dim
        self.contact_joints = opt.contact_joints

        self.layers = nn.ModuleList()
        for i in range(self.joints_num):
            input_dim=512
            input_dim_mid = 256
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, input_dim_mid),
                get_activation(opt.activation),
                nn.Linear(input_dim_mid, self.latent_dim),
            ))

    def forward(self, x):
        """
        x: [bs, nframes, pose_dim]
        
        nfeats = 12J + 1
            - root_rot_velocity (B, seq_len, 1)
            - root_linear_velocity (B, seq_len, 2)
            - root_y (B, seq_len, 1)
            - ric_data (B, seq_len, (joint_num - 1)*3)
            - rot_data (B, seq_len, (joint_num - 1)*6)
            - local_velocity (B, seq_len, joint_num*3)
            - foot contact (B, seq_len, 4)
        """
        B, T, J, D = x.size()   ## 本来是 B, T, 263
        joints = [x[:,:,joint_id] for joint_id in range(0, J)]

        # encode
        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](joints[i]))
        out = torch.stack(out, dim=2)   # [bs, T=64, 32]->[bs, T, 17, 32]

        return out


class MotionDecoder(nn.Module):
    def __init__(self, opt):
        super(MotionDecoder, self).__init__()
        
        self.pose_dim = opt.pose_dim
        # self.joints_num = (self.pose_dim + 1) // 12
        self.joints_num = 17
        self.latent_dim = opt.latent_dim
        self.contact_joints = opt.contact_joints

        # network components
        self.layers = nn.ModuleList()
        for i in range(self.joints_num):
            # if i == 0:
            #     output_dim = 7
            # elif i in self.contact_joints:
            #     output_dim = 13
            # else:
            #     output_dim = 12
            output_dim=512
            latent_dim_mid=256
            self.layers.append(nn.Sequential(
                nn.Linear(self.latent_dim, latent_dim_mid),
                get_activation(opt.activation),
                nn.Linear(latent_dim_mid, output_dim),
            ))

    def forward(self, x):
        """
        x: [bs, nframes, joints_num, latent_dim]
        """
        B, T, J, D = x.size()
        
        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](x[:, :, i]))
        
        motion = torch.stack(out, dim =-2)

        return motion


class STConvEncoder(nn.Module):
    def __init__(self, opt):
        super(STConvEncoder, self).__init__()

        # adjacency list
        # self.adj_list = {
        #     "t2m": t2m_adj_list,
        #     "kit": kit_adj_list,
        # }[opt.dataset_name]
        self.adj_list = h36m_adj_list

        # topology
        self.edge_list = [adj_list_to_edges(self.adj_list)] ## 这个其实是把树形结构转变成俩俩的pair类型
        self.mapping_list = []

        # network
        self.layers = nn.ModuleList()
        for i in range(opt.n_layers):   ## 2
            layers = []
            for _ in range(opt.n_extra_layers):
                layers.append(ResSTConv(
                    self.edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout
                ))
            layers.append(ResSTConv(
                self.edge_list[-1],
                opt.latent_dim,
                opt.kernel_size,
                activation=opt.activation,
                norm=opt.norm,
                dropout=opt.dropout
            ))

            pool = STPool(opt.dataset_name, i)
            layers.append(pool)
            self.layers.append(nn.Sequential(*layers))

            self.edge_list.append(pool.new_edges)
            self.mapping_list.append(pool.skeleton_mapping)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class STConvDecoder(nn.Module):
    def __init__(self, opt, encoder: STConvEncoder):
        super(STConvDecoder, self).__init__()

        # network modules
        self.layers = nn.ModuleList()

        # build network
        mapping_list = encoder.mapping_list.copy()
        edge_list = encoder.edge_list.copy()

        for i in range(opt.n_layers):
            layers = []

            # unpooling
            layers.append(STUnpool(skeleton_mapping=mapping_list.pop()))

            # conv
            edges = edge_list.pop()
            for _ in range(opt.n_extra_layers):
                layers.append(ResSTConv(
                    edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout
                ))
            layers.append(ResSTConv(
                edge_list[-1],
                opt.latent_dim,
                opt.kernel_size,
                activation=opt.activation,
                norm=opt.norm,
                dropout=opt.dropout
            ))

            self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        x: [B, T, J_in, D]
        out: [B, T, J_out, D]
        """
        for layer in self.layers:
            x = layer(x)
        return x