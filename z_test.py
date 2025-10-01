import numpy as np

data = np.load("./debug/train/train_batch_0.npz", allow_pickle=True)
print("db_key:", data['db_key'])  # 输出如 ['00001' '00002']
print("Type:", type(data['db_key']))  # <class 'numpy.ndarray'>
print("Item type:", data['db_key'].dtype)  # object (因为是字符串)

def compare_db_keys(file1, file2):
    a = np.load(file1)['db_key']
    b = np.load(file2)['db_key']
    if np.array_equal(a, b):
        print("✅ db_key: Match")
    else:
        print("❌ db_key: Mismatch")
        print("Train:", a)
        print("Test: ", b)

compare_db_keys("./debug/train/batch_0.npz", "./debug/test/batch_0.npz")
