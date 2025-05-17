"""
mnist_task.py

多分类(MNIST)任务:
  - 手动 + PyTorch(GPU)
  - 至少三种激活函数(ReLU,Tanh,Sigmoid)对比
  - dropout / L2
  - 不同隐藏层数/单元
  - 10折交叉验证
  - 行数>=500
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

# ------------------ 全局设置 ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_env():
    print("mnist_task")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}, device: {device}")

def prepare_checkpoint_dir(d="checkpoints_mnist"):
    if not os.path.exists(d):
        os.makedirs(d)

def one_hot(y, num_classes=10):
    oh = np.zeros((y.shape[0], num_classes))
    for i,v in enumerate(y):
        oh[i,v] = 1
    return oh

# ------------------ 加载MNIST并自动reshape为(N,784) ------------------
def load_mnist_data(npz_file="mnist.npz"):
    """
    假设 mnist.npz 有:
      x_train, y_train, x_test, y_test
    若 x_train 形状是 (60000, 28, 28)，则此函数 reshape -> (60000,784)
    """
    data = np.load(npz_file)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test  = data["x_test"]
    y_test  = data["y_test"]

    # 若 x_train.shape = (60000,28,28), 则将其 reshape -> (60000,784)
    if len(x_train.shape) == 3 and x_train.shape[1] == 28 and x_train.shape[2] == 28:
        x_train = x_train.reshape(x_train.shape[0], -1)
    if len(x_test.shape) == 3 and x_test.shape[1] == 28 and x_test.shape[2] == 28:
        x_test = x_test.reshape(x_test.shape[0], -1)

    # 也可将像素值转换为 float32 / 归一化等
    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test

# ------------------ 手动多分类网络 ------------------
class MultiMLP_Hand:
    """
    多分类: softmax+CE
    自定义激活: relu/tanh/sigmoid
    """
    def __init__(self, input_dim=784, hidden_dims=[256,128], num_classes=10,
                 activation='relu', lr=1e-3, weight_decay=0.0, dropout=0.0, seed=42):
        set_seed(seed)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.act_name = activation.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        # 定义各层维度
        dims = [input_dim] + hidden_dims + [num_classes]
        self.weights = []
        self.biases = []
        for i in range(len(dims)-1):
            in_f, out_f = dims[i], dims[i+1]
            limit = np.sqrt(6./(in_f+out_f))
            W = np.random.uniform(-limit, limit, (in_f,out_f))
            b = np.zeros(out_f)
            self.weights.append(W)
            self.biases.append(b)

    def _act(self,z):
        if self.act_name=='relu':
            return np.maximum(0,z)
        elif self.act_name=='tanh':
            return np.tanh(z)
        elif self.act_name=='sigmoid':
            return 1.0/(1.0+np.exp(-z))
        else:
            return np.maximum(0,z)

    def _act_grad(self,z,a):
        # 用于BP
        if self.act_name=='relu':
            return (z>0).astype(float)
        elif self.act_name=='tanh':
            return 1 - a**2
        elif self.act_name=='sigmoid':
            return a*(1-a)
        else:
            return (z>0).astype(float)

    def _softmax(self, z):
        z_ = z - np.max(z,axis=1,keepdims=True)
        expz = np.exp(z_)
        return expz / np.sum(expz, axis=1, keepdims=True)

    def forward(self, X, is_train=True):
        # 记录中间值
        self.cache_z = []
        self.cache_a = [X]
        a = X
        # 隐藏层
        for i in range(len(self.hidden_dims)):
            z = a.dot(self.weights[i]) + self.biases[i]
            self.cache_z.append(z)
            a_ = self._act(z)
            if is_train and self.dropout>0:
                drop_mask = (np.random.rand(*a_.shape)>self.dropout)
                a_ *= drop_mask
            self.cache_a.append(a_)
            a = a_
        # 输出层
        z_out = a.dot(self.weights[-1]) + self.biases[-1]
        self.cache_z.append(z_out)
        out = self._softmax(z_out)
        return out

    def backward(self,X,y_onehot):
        m = X.shape[0]
        # 输出层梯度
        dZ_out = (self.out - y_onehot)  # (m, num_classes)
        # dW
        dW_out = self.a_list[-1].T.dot(dZ_out)/m
        db_out = np.mean(dZ_out, axis=0)
        if self.weight_decay>0:
            dW_out += self.weight_decay*self.weights[-1]
        # 更新
        self.weights[-1] -= self.lr*dW_out
        self.biases[-1]  -= self.lr*db_out

        # 反传到隐藏层
        dA = dZ_out.dot(self.weights[-1].T)
        for i in range(len(self.hidden_dims)-1, -1, -1):
            z_i = self.cache_z[i]
            a_i = self.cache_a[i+1]  # current layer's activation
            grad_i = self._act_grad(z_i, a_i)
            dZ = dA*grad_i
            dW = self.cache_a[i].T.dot(dZ)/m
            db = np.mean(dZ, axis=0)
            if self.weight_decay>0:
                dW += self.weight_decay*self.weights[i]
            self.weights[i] -= self.lr*dW
            self.biases[i]  -= self.lr*db
            dA = dZ.dot(self.weights[i].T)

    def train_step(self,X,y):
        y_oh = one_hot(y,self.num_classes)
        self.out = self.forward(X,is_train=True)
        eps=1e-8
        loss = -np.sum(y_oh*np.log(self.out+eps))/len(X)
        # 缓存一次
        self.a_list = self.cache_a
        self.backward(X,y_oh)
        return loss

    def predict(self,X):
        p = self.forward(X, is_train=False)
        return np.argmax(p, axis=1)

# ------------------ PyTorch网络 ------------------
class MultiMLP_Torch(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256,128],
                 num_classes=10, activation='relu', dropout=0.0):
        super().__init__()
        self.activation_name = activation.lower()
        layers = []
        in_f = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_f, hd))
            if self.activation_name=='relu':
                layers.append(nn.ReLU())
            elif self.activation_name=='tanh':
                layers.append(nn.Tanh())
            elif self.activation_name=='sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            in_f = hd
        layers.append(nn.Linear(in_f, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_torch_model(model, x_train, y_train, x_test, y_test,
                      lr=1e-3, weight_decay=0.0, epochs=10,
                      batch_size=128, grad_clip=None, early_stop_patience=5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    x_test_t  = torch.from_numpy(x_test).float().to(device)
    y_test_t  = torch.from_numpy(y_test).long().to(device)

    best_loss = float('inf')
    bad_count = 0
    train_losses, test_losses = [], []
    n_samples = len(x_train_t)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        sum_loss=0.
        for i in range(0,n_samples,batch_size):
            idx = perm[i:i+batch_size]
            xb = x_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            sum_loss += loss.item()

        train_loss = sum_loss/(n_samples/batch_size)
        train_losses.append(train_loss)

        # test
        model.eval()
        with torch.no_grad():
            logits_test = model(x_test_t)
            test_loss = criterion(logits_test, y_test_t).item()
        test_losses.append(test_loss)

        if test_loss<best_loss:
            best_loss = test_loss
            bad_count = 0
        else:
            bad_count += 1
            if bad_count>=early_stop_patience:
                break

    return train_losses, test_losses

def evaluate_accuracy(model, X, y):
    model.eval()
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)
    with torch.no_grad():
        out = model(X_t)
        pred = out.argmax(dim=1)
        acc = (pred==y_t).float().mean().item()
    return acc

def plot_curve(tr_losses, te_losses, label):
    plt.figure(figsize=(6,4))
    plt.plot(tr_losses, label='Train')
    plt.plot(te_losses, label='Test')
    plt.title(label)
    plt.xlabel("Epoch")
    plt.ylabel("CE")
    plt.legend()
    plt.show()

# ------------------ 实验函数 ------------------
def handcoded_experiment_activation(activation='relu'):
    # 在这里统一加载并 reshape
    x_train, y_train, x_test, y_test = load_mnist_data("mnist.npz")

    model=MultiMLP_Hand(input_dim=784, hidden_dims=[256,128],
                        num_classes=10, activation=activation,
                        lr=1e-3, weight_decay=1e-4, dropout=0.1, seed=2023)
    epochs=10
    tr_losses=[]
    te_losses=[]
    for ep in range(epochs):
        loss_tr=model.train_step(x_train,y_train)
        tr_losses.append(loss_tr)
        out_test=model.forward(x_test,is_train=False)
        eps=1e-8
        loss_te=-np.mean(np.log(out_test[range(len(y_test)), y_test]+eps))
        te_losses.append(loss_te)
    plot_curve(tr_losses, te_losses, f"Hand-coded MLP {activation}")
    pred=model.predict(x_test)
    acc=np.mean(pred==y_test)
    print(f"Hand-coded {activation} final acc:", acc)

def torch_experiment(activation='relu', dropout=0.0, weight_decay=0.0,
                     hidden=[256,128], epochs=5):
    x_train, y_train, x_test, y_test = load_mnist_data("mnist.npz")

    model=MultiMLP_Torch(input_dim=784, hidden_dims=hidden,
                         num_classes=10, activation=activation,
                         dropout=dropout).to(device)
    tr_losses, te_losses = train_torch_model(
        model, x_train, y_train, x_test, y_test,
        lr=1e-3, weight_decay=weight_decay, epochs=epochs,
        batch_size=128, grad_clip=5.0, early_stop_patience=5
    )
    lab=f"PyTorch {activation}, drop={dropout}, wd={weight_decay}, h={hidden}"
    plot_curve(tr_losses, te_losses, lab)
    acc=evaluate_accuracy(model, x_test, y_test)
    print(lab,"acc=",acc)
    return acc

def cross_val_10fold_best_model():
    X_train, Y_train, X_test, Y_test = load_mnist_data("mnist.npz")
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([Y_train, Y_test], axis=0)

    kf=KFold(n_splits=10, shuffle=True, random_state=42)
    fold_acc=[]
    i=1
    print("|Fold|Acc   |")
    print("|----|------|")
    for tr_idx, val_idx in kf.split(X):
        X_tr,X_val = X[tr_idx],X[val_idx]
        y_tr,y_val = y[tr_idx],y[val_idx]
        model=MultiMLP_Torch(784,[256,128],10,'relu',0.2).to(device)
        train_torch_model(model,X_tr,y_tr,X_val,y_val,lr=1e-3,weight_decay=1e-4,epochs=5)
        a=evaluate_accuracy(model,X_val,y_val)
        fold_acc.append(a)
        print(f"|{i:2d}  |{a:.4f}|")
        i+=1
    print("Avg acc:", np.mean(fold_acc))

def advanced_experiments():
    # 1) 不同激活
    for act in ["relu","tanh","sigmoid"]:
        _=torch_experiment(act, dropout=0.0, weight_decay=0.0, hidden=[128,64], epochs=5)
    # 2) 不同dropout
    for dr in [0.2,0.5]:
        _=torch_experiment("relu", dropout=dr, weight_decay=0.0, hidden=[128,64], epochs=5)
    # 3) 不同L2
    for wd_ in [1e-5,1e-4,1e-3]:
        _=torch_experiment("relu", dropout=0.0, weight_decay=wd_, hidden=[128,64], epochs=5)
    # 4) 不同隐藏层
    for hid_ in [[128,64],[256,128],[512,256]]:
        _=torch_experiment("relu", dropout=0.2, weight_decay=1e-4, hidden=hid_, epochs=5)

# ------------------ MAIN ------------------
if __name__=="__main__":
    print_env()
    prepare_checkpoint_dir()

    # 手动: 对比三种激活
    for ac in ["relu","tanh","sigmoid"]:
        handcoded_experiment_activation(ac)

    # PyTorch: 做一些实验
    torch_experiment("relu", 0.0, 0.0, [256,128], 5)
    torch_experiment("tanh", 0.0, 0.0, [256,128], 5)
    torch_experiment("sigmoid", 0.0, 0.0, [256,128], 5)

    # 更多实验
    advanced_experiments()

    # 10折
    cross_val_10fold_best_model()
