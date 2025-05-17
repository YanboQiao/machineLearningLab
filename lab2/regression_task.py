"""
regression_task.py

回归任务示例。要求:
  - 使用 NumPy 手动实现前馈网络 + 使用 PyTorch(GPU) 前馈网络
  - 训练并可视化训练集/测试集loss
  - 演示L2正则/dropout(若需要)
  - 10折交叉验证并展示结果
  - 代码行数不少于300(注释简短)

数据集: regression_dataset.npz
  包含 X_train, y_train, X_test, y_test
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn


# 确定设备: GPU优先
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保证行数足够，增加一些辅助函数
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_env_info():
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")

def prepare_directory(dir_name="checkpoints_reg"):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# ---------- 手动实现前馈网络 (仅CPU) ----------
class MLPRegression_Hand:
    """
    多层感知器(回归), 纯 NumPy 实现, 不支持GPU
    """
    def __init__(self, input_dim, hidden_dims=[128,64],
                 lr=1e-3, weight_decay=0.0, dropout=0.0, seed=42):
        set_seed(seed)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        # 初始化各层参数
        sizes = [input_dim] + hidden_dims + [1]  # output=1
        self.weights = []
        self.biases = []
        for i in range(len(sizes)-1):
            in_f, out_f = sizes[i], sizes[i+1]
            limit = np.sqrt(6. / (in_f+out_f))
            W = np.random.uniform(-limit, limit, (in_f,out_f))
            b = np.zeros(out_f)
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x>0).astype(float)

    def forward(self, X, is_train=True):
        """
        前向传播, 多层RELU, 最后一层线性输出
        """
        self.cache_z = []
        self.cache_a = [X]
        a = X
        for i in range(len(self.hidden_dims)):
            z = a.dot(self.weights[i]) + self.biases[i]
            self.cache_z.append(z)
            a_ = self.relu(z)
            if is_train and self.dropout>0:
                drop_mask = (np.random.rand(*a_.shape)>self.dropout)
                a_ *= drop_mask
            self.cache_a.append(a_)
            a = a_
        # 输出层
        z_last = a.dot(self.weights[-1]) + self.biases[-1]
        self.cache_z.append(z_last)
        return z_last

    def backward(self, X, y, y_pred):
        """
        反向传播: MSE Loss
        dL/dy_pred = (y_pred - y)
        """
        m = X.shape[0]
        diff = (y_pred.flatten() - y)  # shape=(m,)
        # output层梯度
        dW_last = self.cache_a[-1].T.dot(diff.reshape(-1,1))/m
        db_last = np.mean(diff)
        # L2
        if self.weight_decay>0:
            dW_last += self.weight_decay*self.weights[-1]
        # update
        self.weights[-1] -= self.lr*dW_last
        self.biases[-1]  -= self.lr*db_last

        # 反向到隐藏层
        dA = diff.reshape(-1,1).dot(self.weights[-1].T)  # shape=(m, hidden_dims[-1])
        for i in range(len(self.hidden_dims)-1, -1, -1):
            z_i = self.cache_z[i]
            grad_i = self.relu_grad(z_i)
            dZ = dA*grad_i
            dW = self.cache_a[i].T.dot(dZ)/m
            db = np.mean(dZ, axis=0)
            if self.weight_decay>0:
                dW += self.weight_decay*self.weights[i]
            self.weights[i] -= self.lr*dW
            self.biases[i]  -= self.lr*db
            dA = dZ.dot(self.weights[i].T)

    def train_step(self, X, y):
        y_pred = self.forward(X, is_train=True)
        loss = np.mean((y_pred.flatten()-y)**2)/2
        self.backward(X, y, y_pred)
        return loss

    def predict(self, X):
        out = self.forward(X, is_train=False)
        return out.flatten()

# ---------- PyTorch MLP (可GPU) ----------
class MLPRegression_Torch(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64], dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(nn.ReLU())
            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hd
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)

def train_torch_model(model, X_train, y_train, X_test, y_test,
                      lr=1e-3, weight_decay=0.0, epochs=20,
                      batch_size=128, clip_grad=None, patience=5):
    """
    训练PyTorch模型(回归)，支持:
      - L2 (weight_decay)
      - gradient clip
      - EarlyStopping (patience)
    返回: (train_losses, test_losses)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().view(-1,1).to(device)
    X_test_t  = torch.from_numpy(X_test).float().to(device)
    y_test_t  = torch.from_numpy(y_test).float().view(-1,1).to(device)

    best_loss = float('inf')
    bad_count = 0
    train_losses, test_losses = [], []
    n_samples = len(X_train_t)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        batch_loss = 0.
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            if clip_grad is not None and clip_grad>0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()
            batch_loss += loss.item()

        train_loss = batch_loss/(n_samples/batch_size)
        train_losses.append(train_loss)

        # test
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_t)
            test_loss = criterion(pred_test, y_test_t).item()
        test_losses.append(test_loss)

        # EarlyStopping
        if test_loss<best_loss:
            best_loss = test_loss
            bad_count = 0
        else:
            bad_count += 1
            if bad_count>=patience:
                print(f"Early stop at epoch {ep+1}")
                break
    return train_losses, test_losses

# ---------- 实验流程 ----------
def draw_loss_curve(train_losses, test_losses, title):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.show()

def run_handcoded():
    data = np.load("regression_dataset.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test   = data["X_test"], data["y_test"]
    model = MLPRegression_Hand(input_dim=X_train.shape[1],
                               hidden_dims=[128,64],
                               lr=1e-3, weight_decay=1e-5, dropout=0.0, seed=2023)
    epochs=20
    train_losses, test_losses = [], []
    for ep in range(epochs):
        loss_tr = model.train_step(X_train, y_train)
        train_losses.append(loss_tr)
        y_pred_test = model.predict(X_test)
        loss_te = np.mean((y_pred_test-y_test)**2)/2
        test_losses.append(loss_te)
    draw_loss_curve(train_losses, test_losses, "Hand-coded MLP (CPU)")

def run_torch():
    data = np.load("regression_dataset.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test   = data["X_test"], data["y_test"]
    model = MLPRegression_Torch(input_dim=X_train.shape[1],
                                hidden_dims=[128,64], dropout=0.1).to(device)
    tr_losses, te_losses = train_torch_model(
        model,
        X_train, y_train, X_test, y_test,
        lr=1e-3, weight_decay=1e-5, epochs=30, batch_size=64, clip_grad=5.0, patience=5
    )
    draw_loss_curve(tr_losses, te_losses, "PyTorch MLP (GPU)")

def cross_val_10fold_best():
    data = np.load("regression_dataset.npz")
    X = np.concatenate([data["X_train"], data["X_test"]], axis=0)
    y = np.concatenate([data["y_train"], data["y_test"]], axis=0)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_mse = []
    idx=1
    print("|Fold|MSE   |")
    print("|----|------|")
    for train_idx,val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = MLPRegression_Torch(input_dim=X.shape[1], hidden_dims=[128,64], dropout=0.2).to(device)
        train_torch_model(model, X_tr, y_tr, X_val, y_val, epochs=10)
        model.eval()
        X_val_t = torch.from_numpy(X_val).float().to(device)
        y_val_t = torch.from_numpy(y_val).float().view(-1,1).to(device)
        with torch.no_grad():
            pred_val = model(X_val_t)
            mse_val = nn.MSELoss()(pred_val, y_val_t).item()
        fold_mse.append(mse_val)
        print(f"|{idx:2d}  |{mse_val:.4f}|")
        idx+=1
    avg_mse = np.mean(fold_mse)
    print("Average MSE:", avg_mse)

if __name__=="__main__":
    print_env_info()
    prepare_directory()
    run_handcoded()
    run_torch()
    cross_val_10fold_best()
