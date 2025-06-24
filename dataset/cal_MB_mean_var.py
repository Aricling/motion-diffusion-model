import numpy as np
# from tqdm import tqdm

# data_dict_all=np.load("/home/mengqing/usr/motion-diffusion-model/dataset/t2m_train.npy", allow_pickle=True)[None][0]
# name_list, length_list, data_dict = data_dict_all['name_list'], data_dict_all['length_list'], data_dict_all['data_dict']

# motion_emb_list=[]
# for i, name in enumerate(name_list):
#     data = data_dict[name]
#     motion, m_length, text_list, motion_emb = data['motion'], data['length'], data['text'], data["motion_emb"]
#     print(motion.shape, m_length, motion_emb.shape)
#     motion_emb_list.append(motion_emb)
# arr = np.concatenate(motion_emb_list)
# motion_emb_mean=arr.mean(axis=0)
# motion_emb_std = arr.std(axis=0)
# print(motion_emb_mean.shape, motion_emb_std.shape)
# np.save("/home/mengqing/usr/motion-diffusion-model/dataset/motion_emb_mean.npy", motion_emb_mean)
# np.save("/home/mengqing/usr/motion-diffusion-model/dataset/motion_emb_std.npy", motion_emb_std)

mean=np.load("/home/mengqing/usr/motion-diffusion-model/dataset/motion_emb_mean.npy")
std=np.load("/home/mengqing/usr/motion-diffusion-model/dataset/motion_emb_std.npy")

# 保存 mean 到文本文件
with open("/home/mengqing/usr/z_old/motion_emb_mean.txt", "w") as f:
    f.write("mean shape: " + str(mean.shape) + "\n")
    np.savetxt(f, mean, fmt="%.6f")  # 保留6位小数

# 保存 std 到文本文件
with open("/home/mengqing/usr/z_old/motion_emb_std.txt", "w") as f:
    f.write("std shape: " + str(std.shape) + "\n")
    np.savetxt(f, std, fmt="%.6f")  # 保留6位小数

print("Saved mean and std to motion_emb_mean.txt and motion_emb_std.txt")