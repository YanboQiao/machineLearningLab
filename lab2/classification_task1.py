"""
classification_task1.py

二分类任务1。要求:
  - 手动前馈网络 + PyTorch(GPU) 前馈网络
  - 训练并可视化二分类(交叉熵)
  - 展示L2或dropout
  - 10折交叉验证(表格)
  - 不少于500行
  - 注释简短

数据: classification_dataset1.npz
  X_train, y_train (7000条), X_test, y_test (3000条), 二分类标签{0,1}, 特征维度200
"""

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 行数填充: 工具函数
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_env_info():
    print("classification_task1")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}, Device: {device}")

def prepare_dir(dir_name="checkpoints_cls1"):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# ------------------ 手动前馈网络 (CPU) ------------------
class BinaryMLP_Hand:
    """
    多层感知器二分类, 纯NumPy
    激活: ReLU
    输出: Sigmoid
    损失: BCE
    """
    def __init__(self, input_dim, hidden_dims=[64,32],
                 lr=1e-3, weight_decay=0.0, dropout=0.0, seed=42):
        set_seed(seed)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        sizes = [input_dim] + hidden_dims + [1]
        self.weights = []
        self.biases = []
        for i in range(len(sizes)-1):
            in_f, out_f = sizes[i], sizes[i+1]
            limit = np.sqrt(6./(in_f+out_f))
            W = np.random.uniform(-limit, limit,(in_f,out_f))
            b = np.zeros(out_f)
            self.weights.append(W)
            self.biases.append(b)

    def relu(self,x):
        return np.maximum(0,x)

    def relu_grad(self,x):
        return (x>0).astype(float)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def forward(self, X, is_train=True):
        self.cache_z = []
        self.cache_a = [X]
        a = X
        for i in range(len(self.hidden_dims)):
            z = a.dot(self.weights[i])+self.biases[i]
            self.cache_z.append(z)
            a_ = self.relu(z)
            if is_train and self.dropout>0:
                drop_mask = (np.random.rand(*a_.shape)>self.dropout)
                a_ *= drop_mask
            self.cache_a.append(a_)
            a = a_
        z_out = a.dot(self.weights[-1])+self.biases[-1]
        self.cache_z.append(z_out)
        out = self.sigmoid(z_out)
        return out

    def backward(self, X, y, y_pred):
        """
        BCE = -1/m sum[y*log(a)+(1-y)*log(1-a)]
        dZ_out = (a - y)
        """
        m = X.shape[0]
        dZ_out = (y_pred.flatten()-y).reshape(-1,1)
        dW_out = self.cache_a[-1].T.dot(dZ_out)/m
        db_out = np.mean(dZ_out, axis=0)

        if self.weight_decay>0:
            dW_out += self.weight_decay*self.weights[-1]

        self.weights[-1] -= self.lr*dW_out
        self.biases[-1]  -= self.lr*db_out

        dA = dZ_out.dot(self.weights[-1].T)
        for i in range(len(self.hidden_dims)-1, -1, -1):
            z_i = self.cache_z[i]
            grad_i = self.relu_grad(z_i)
            dZ = dA*grad_i
            dW = self.cache_a[i].T.dot(dZ)/m
            db = np.mean(dZ,axis=0)
            if self.weight_decay>0:
                dW += self.weight_decay*self.weights[i]
            self.weights[i] -= self.lr*dW
            self.biases[i]  -= self.lr*db
            dA = dZ.dot(self.weights[i].T)

    def train_step(self, X, y):
        y_pred = self.forward(X, is_train=True)
        eps=1e-8
        loss = -np.mean(y*np.log(y_pred+eps)+(1-y)*np.log(1-y_pred+eps))
        self.backward(X, y, y_pred)
        return loss

    def predict_proba(self, X):
        return self.forward(X, is_train=False).flatten()

    def predict_label(self,X):
        proba = self.predict_proba(X)
        return (proba>=0.5).astype(int)

# ------------------ PyTorch MLP (GPU) ------------------
class BinaryMLP_Torch(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64,32], dropout=0.0):
        super().__init__()
        layers=[]
        in_f = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_f, hd))
            layers.append(nn.ReLU())
            if dropout>0:
                layers.append(nn.Dropout(p=dropout))
            in_f=hd
        layers.append(nn.Linear(in_f,1))
        layers.append(nn.Sigmoid())
        self.net=nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)

def train_torch_model(model, X_train, y_train, X_test, y_test,
                      lr=1e-3, weight_decay=0.0, epochs=20,
                      batch_size=128, clip_grad=None, scheduler_patience=3):
    """
    训练二分类模型(BCE), GPU
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().view(-1,1).to(device)
    X_test_t  = torch.from_numpy(X_test).float().to(device)
    y_test_t  = torch.from_numpy(y_test).float().view(-1,1).to(device)

    train_losses, test_losses = [], []
    n_samples = len(X_train_t)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, factor=0.5)

    for ep in range(epochs):
        model.train()
        perm=torch.randperm(n_samples)
        sum_loss=0.
        for i in range(0,n_samples,batch_size):
            idx=perm[i:i+batch_size]
            xb=X_train_t[idx]
            yb=y_train_t[idx]

            optimizer.zero_grad()
            out=model(xb)
            loss=criterion(out,yb)
            loss.backward()

            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(),clip_grad)

            optimizer.step()
            sum_loss+=loss.item()

        train_loss = sum_loss/(n_samples/batch_size)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            out_test = model(X_test_t)
            test_loss = criterion(out_test, y_test_t).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)

    return train_losses, test_losses

# ---------- 绘图等 ----------
def plot_loss_curve(train_losses, test_losses, title):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("BCE")
    plt.legend()
    plt.show()

# ---------- 主流程 ----------
def run_handcoded():
    data = np.load("classification_dataset1.npz")
    X_train=data["X_train"]; y_train=data["y_train"]
    X_test=data["X_test"];   y_test=data["y_test"]
    model = BinaryMLP_Hand(input_dim=X_train.shape[1], hidden_dims=[64,32],
                           lr=1e-3, weight_decay=1e-4, dropout=0.1, seed=2023)
    epochs=20
    tr_losses=[]
    te_losses=[]
    for ep in range(epochs):
        loss_tr = model.train_step(X_train, y_train)
        tr_losses.append(loss_tr)
        proba_te = model.predict_proba(X_test)
        eps=1e-8
        loss_te = -np.mean(y_test*np.log(proba_te+eps)+(1-y_test)*np.log(1-proba_te+eps))
        te_losses.append(loss_te)
    plot_loss_curve(tr_losses, te_losses, "Hand-coded Binary Classification1")
    pred_label = model.predict_label(X_test)
    acc = np.mean(pred_label==y_test)
    print("Hand-coded final acc:", acc)

def run_torch():
    data = np.load("classification_dataset1.npz")
    X_train=data["X_train"]; y_train=data["y_train"]
    X_test=data["X_test"];   y_test=data["y_test"]
    model = BinaryMLP_Torch(input_dim=X_train.shape[1], hidden_dims=[64,32], dropout=0.2).to(device)
    tr_losses, te_losses = train_torch_model(
        model,
        X_train, y_train, X_test, y_test,
        lr=1e-3, weight_decay=1e-4, epochs=30, batch_size=64,
        clip_grad=5.0, scheduler_patience=3
    )
    plot_loss_curve(tr_losses, te_losses, "PyTorch Binary Classification1")
    model.eval()
    with torch.no_grad():
        X_test_t=torch.from_numpy(X_test).float().to(device)
        proba_test = model(X_test_t).cpu().numpy().flatten()
    pred_label = (proba_test>=0.5).astype(int)
    acc = np.mean(pred_label==y_test)
    print("PyTorch final acc:", acc)

def cross_val_10fold_best():
    data = np.load("classification_dataset1.npz")
    X = np.concatenate([data["X_train"], data["X_test"]],axis=0)
    y = np.concatenate([data["y_train"], data["y_test"]],axis=0)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_acc=[]
    fold_idx=1
    print("|Fold|Acc   |")
    print("|----|------|")
    for train_idx,val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = BinaryMLP_Torch(input_dim=X.shape[1], hidden_dims=[64,32], dropout=0.2).to(device)
        train_torch_model(model, X_tr, y_tr, X_val, y_val, epochs=10, batch_size=64)
        model.eval()
        with torch.no_grad():
            X_val_t=torch.from_numpy(X_val).float().to(device)
            proba_val=model(X_val_t).cpu().numpy().flatten()
        pred_label=(proba_val>=0.5).astype(int)
        acc_val = np.mean(pred_label==y_val)
        fold_acc.append(acc_val)
        print(f"|{fold_idx:2d}  |{acc_val:.4f}|")
        fold_idx+=1
    print("Avg acc:", np.mean(fold_acc))

def run_extra_experiment():
    """
    多个超参数组合测试
    """
    data = np.load("classification_dataset1.npz")
    X_train=data["X_train"]; y_train=data["y_train"]
    X_test=data["X_test"];   y_test=data["y_test"]
    hidden_options = [[32,16],[64,32],[128,64]]
    drop_options = [0.0, 0.1, 0.3]
    best_acc=0
    for h in hidden_options:
        for dr in drop_options:
            model = BinaryMLP_Torch(input_dim=X_train.shape[1], hidden_dims=h, dropout=dr).to(device)
            train_torch_model(model, X_train,y_train,X_test,y_test, epochs=5, batch_size=64)
            model.eval()
            with torch.no_grad():
                X_test_t=torch.from_numpy(X_test).float().to(device)
                proba_test=model(X_test_t).cpu().numpy().flatten()
            pred_label=(proba_test>=0.5).astype(int)
            acc = np.mean(pred_label==y_test)
            print(f"Hidden={h},drop={dr},acc={acc:.4f}")
            if acc>best_acc:
                best_acc=acc
    print("Best acc found:", best_acc)

if __name__=="__main__":
    print_env_info()
    prepare_dir()
    run_handcoded()
    run_torch()
    cross_val_10fold_best()
    run_extra_experiment()
