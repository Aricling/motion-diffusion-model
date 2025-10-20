from torch.utils.data import Sampler, BatchSampler, DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_prefix_collate
import z_config

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0, 
                device=None, autoregressive=False, cache_path=None): 
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
                       device=device, autoregressive=autoregressive)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0, 
                       device=None, autoregressive=False):
    dataset = get_dataset(name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len, 
                device=device, autoregressive=autoregressive)
    
    collate = get_collate_fn(name, hml_mode, pred_len, batch_size)

    if z_config.get_diy_config().debug:
        your_indices = [947, 
            4186, 2944, 1946, 816, 2007,
            518, 539, 1505, 3317, 2515,
            2845, 795, 437, 541, 683,
            3040, 2937, 1414, 1133, 1317,
            733, 3382, 1311, 411, 1333,
            169, 3259, 1438, 1377, 4141,
            982
        ]

        # 自定义 Sampler：只返回你指定的索引
        class SpecificIndicesSampler(Sampler):
            def __init__(self, indices):
                self.indices = indices

            def __iter__(self):
                return iter(self.indices)

            def __len__(self):
                return len(self.indices)

        # 创建 Sampler 和 BatchSampler
        sampler = SpecificIndicesSampler(your_indices)

        # 使用 BatchSampler 将所有索引打包成一个 batch
        # 注意：drop_last=False 确保即使 batch_size > len(indices) 也会返回
        batch_sampler = BatchSampler(sampler, batch_size=len(your_indices), drop_last=False)
        ###########
        
        loader = DataLoader(
            dataset,  # 你的数据集
            batch_sampler=batch_sampler,
            num_workers=0,  # 建议设为 0 以避免多进程干扰
            collate_fn=collate,  # 如果你有自定义的 collate 函数，记得传入
        ) 
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=8, drop_last=True, collate_fn=collate
        )

    return loader