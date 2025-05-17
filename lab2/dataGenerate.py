import numpy as np

def generate_regression_data(
    N=10000,          # 总样本量
    p=500,            # 特征维度
    train_size=7000,  # 训练集大小
    noise_std=1.0,    # 回归噪声标准差
    random_seed=42
):
    """
    回归任务数据生成:
    y = 0.028 + sum(0.0056 * x_i) + eps
    其中 eps ~ N(0, noise_std^2).
    """

    np.random.seed(random_seed)

    # 生成特征 X
    X = np.random.randn(N, p)  # 标准正态分布

    # 生成标签 y
    # 线性部分: 0.028 + 0.0056 * sum(x_i)
    # 噪声部分: eps ~ N(0, noise_std^2)
    linear_part = 0.028 + 0.0056 * np.sum(X, axis=1)
    eps = noise_std * np.random.randn(N)
    y = linear_part + eps

    # 划分训练集和测试集
    indices = np.random.permutation(N)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test   = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test

def generate_classification_data(
    N=10000,           # 总样本量
    p=200,             # 特征维度
    train_size=7000,   # 训练集大小
    mu_val=2.0,        # |均值| 值
    random_seed=42
):
    """
    二分类任务数据生成:
    - 标签 0: X ~ N(-mu, I)
    - 标签 1: X ~ N(+mu, I)
    其中 mu 是一个长度为 p 的常数向量 [mu_val, mu_val, ..., mu_val].
    """

    np.random.seed(random_seed)

    # 每类样本各一半
    N_class0 = N // 2
    N_class1 = N - N_class0

    # 定义均值向量
    mu = mu_val * np.ones(p)
    neg_mu = -mu

    # 生成标签 0 的样本
    X0 = np.random.randn(N_class0, p) + neg_mu  # N(-mu, I)
    y0 = np.zeros(N_class0, dtype=int)

    # 生成标签 1 的样本
    X1 = np.random.randn(N_class1, p) + mu      # N(+mu, I)
    y1 = np.ones(N_class1, dtype=int)

    # 合并并打乱
    X_all = np.concatenate([X0, X1], axis=0)
    y_all = np.concatenate([y0, y1], axis=0)

    indices = np.random.permutation(N)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test   = X_all[test_idx], y_all[test_idx]

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # 1. 生成回归数据
    X_train_reg, y_train_reg, X_test_reg, y_test_reg = generate_regression_data(
        N=10000,
        p=500,
        train_size=7000,
        noise_std=1.0,
        random_seed=42
    )
    # 保存回归数据
    np.savez("regression_dataset.npz",
             X_train=X_train_reg, y_train=y_train_reg,
             X_test=X_test_reg, y_test=y_test_reg)
    print("回归数据已生成并保存到 regression_dataset.npz")

    # 2. 生成第一个二分类数据集
    X_train_cls1, y_train_cls1, X_test_cls1, y_test_cls1 = generate_classification_data(
        N=10000,
        p=200,
        train_size=7000,
        mu_val=2.0,
        random_seed=42
    )
    np.savez("classification_dataset1.npz",
             X_train=X_train_cls1, y_train=y_train_cls1,
             X_test=X_test_cls1, y_test=y_test_cls1)
    print("第一个二分类数据集已生成并保存到 classification_dataset1.npz")

    # 3. 生成第二个二分类数据集（可换一个随机种子或换 mu_val）
    X_train_cls2, y_train_cls2, X_test_cls2, y_test_cls2 = generate_classification_data(
        N=10000,
        p=200,
        train_size=7000,
        mu_val=2.0,  # 也可改成其他值, 如 3.0, 4.0 等
        random_seed=2023
    )
    np.savez("classification_dataset2.npz",
             X_train=X_train_cls2, y_train=y_train_cls2,
             X_test=X_test_cls2, y_test=y_test_cls2)
    print("第二个二分类数据集已生成并保存到 classification_dataset2.npz")
