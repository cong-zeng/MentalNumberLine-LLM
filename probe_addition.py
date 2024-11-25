import torch
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
# 使用DataLoader
from torch.utils.data import TensorDataset, DataLoader
import wandb


WANDB_API_KEY="532007cd7a07c1aa0d1194049c3231dadd1d418e"
# Name and notes optional
wandb.login(key=WANDB_API_KEY)

EPOCHS = 801
TEST_SIZE = 0.6
VAL_SIZE = 0.5
LEARNING_RATE = 0.001
HIDDEN_PARAM = [1024, 1024, 1024]
NG_SLOPE = 0.01
BATCH_SIZE = 2048
RAND_SEED_1 = 42
RAND_SEED_2 = 43
IS_BATCH_NORM = False
IS_LAYER_NORM = True


def make_multi_layers(input_dim, output_dim, hidden_param, is_batch_norm=False, is_layer_norm=False):
    """
    Creates a neural network with multiple layers, allowing optional BatchNorm or LayerNorm.
    
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output layer.
        hidden_param (list): List of hidden layer sizes.
        is_batch_norm (bool): If True, apply BatchNorm after each layer.
        is_layer_norm (bool): If True, apply LayerNorm after each layer.
        
    Returns:
        nn.Sequential: The constructed neural network as a sequential model.
    """
    def add_layer(layers, in_dim, out_dim):
        layers.append(nn.Linear(in_dim, out_dim))
        if is_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        if is_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.LeakyReLU(negative_slope=NG_SLOPE))

    layers = []
    # Add first layer
    add_layer(layers, input_dim, hidden_param[0])
    # Add hidden layers
    for i in range(1, len(hidden_param)):
        add_layer(layers, hidden_param[i-1], hidden_param[i])
    # Add output layer (no activation or normalization)
    layers.append(nn.Linear(hidden_param[-1], output_dim))
    
    return nn.Sequential(*layers)
        

# 定义一个函数，用于生成求和的数据集
def generate_addition_dataset(embeddings, labels, is_reduce_dim=False):
    """
    生成两数求和的数据集。

    参数：
    embeddings: (200, d) 的Tensor，代表200个embedding
    labels: (200,) 的Tensor，代表每个embedding对应的数字标签
    is_reduce_dim: 是否对embedding降维

    返回：
    features: (num_pairs, 2*d) 的Tensor，每个样本由两个embedding拼接而成
    targets: (num_pairs,) 的Tensor，两数之和的标签
    """
    if is_reduce_dim:
        # 降维
        embeddings = TSNE(n_components=2, perplexity=20).fit_transform(embeddings)
    # Convert embeddings to PyTorch Tensor
    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    features = []
    targets = []

    emb_num = len(embeddings)
    print(f"emb_num: {emb_num}")
    max_addition = 2 * max(labels)
    print(f"max_addition: {max_addition}")
    truncated_num = int(max_addition / 4)
    print(f"truncated_num: {truncated_num}")

    for i in range(emb_num):
        for j in range(emb_num):
            # 生成两数之和的标签
            label_1 = labels[i]
            label_2 = labels[j]
            target = label_1 + label_2
            if target < 2 * min(labels) + truncated_num or target > max_addition - truncated_num:
                continue
            targets.append(target)

            embedding_1 = embeddings[i]
            embedding_2 = embeddings[j]

            # 拼接两个embedding
            combined_feature = torch.cat((embedding_1, embedding_2), dim=0)
            features.append(combined_feature)

    # 转换为Tensor
    features = torch.stack(features)
    targets = torch.tensor(targets, dtype=torch.long)
    # 调整目标标签使其从0开始
    targets = targets - targets.min()

    return features, targets

# 定义一个简单的线性探测器
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = make_multi_layers(
            input_dim, output_dim, HIDDEN_PARAM, 
            is_batch_norm=IS_BATCH_NORM, is_layer_norm=IS_LAYER_NORM)


    def forward(self, x):
        return self.linear(x)

def probe_num_addition_accu(embs, labels, n_emb=201, exp_info=None):
    # 限制 emb 数量
    if len(embs) > n_emb:
        embs = embs[:n_emb]
        labels = labels[:n_emb]

    # 检查CUDA是否可用
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")

    # 生成两数求和的数据集
    features, targets = generate_addition_dataset(embs, labels, is_reduce_dim=False)
    print(f"features shape: {features.shape}, targets shape: {targets.shape}")

    # 划分训练集、验证集和测试集
    X_other, X_test, y_other, y_test = train_test_split(features, targets, test_size=TEST_SIZE, random_state=RAND_SEED_1)
    X_train, X_val, y_train, y_val = train_test_split(X_other, y_other, test_size=VAL_SIZE, random_state=RAND_SEED_2)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 初始化模型、损失函数和优化器
    input_dim = X_train.shape[1]
    num_classes = targets.max().item() + 1
    print(f"num_classes: {num_classes}")
    model = LinearProbe(input_dim, num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if exp_info is not None:
        exp_info["n_classes"] = num_classes
        exp_info["lr"] = LEARNING_RATE
        exp_info["train_size"] = (1-TEST_SIZE)*VAL_SIZE
        exp_info["n_samples"] = len(features)
        exp_info["emb_dim"] = X_train.shape[1]
        exp_info["hidden_param"] = HIDDEN_PARAM
        exp_info["ng_slope"] = NG_SLOPE
        exp_info["max_epoch"] = EPOCHS
        exp_info["batch_size"] = BATCH_SIZE
        exp_info["rand_seed_1"] = RAND_SEED_1
        exp_info["rand_seed_2"] = RAND_SEED_2
        exp_info["batch_norm"] = IS_BATCH_NORM
        exp_info["layer_norm"] = IS_LAYER_NORM
        wandb.init(
            project="MNL-num_addtion",
            name=f"{exp_info['model']}_{exp_info['t_method']}_{exp_info['prompt']}",
            config=exp_info,
            tags=[f"{key}:{value}" for key, value in exp_info.items()],
            group=f"{exp_info['layer']}",
        )

    # 创建TensorDataset和DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 训练模型
    best_val_accu = 0
    exit_epoch = 0
    best_model_state = None

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_X.size(0)

        train_loss = total_loss / total_samples
        train_accu = total_correct / total_samples

        if epoch % 10 == 0:
            # 验证模型
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_X = val_X.to(DEVICE)
                    val_y = val_y.to(DEVICE)
                    val_outputs = model(val_X)
                    val_preds = val_outputs.argmax(dim=1)
                    val_correct += (val_preds == val_y).sum().item()
                    val_total += val_X.size(0)
                    val_loss += criterion(val_outputs, val_y).item() * val_X.size(0)
            val_accu = val_correct / val_total
            val_loss /= val_total
            # print(f"Epoch: {epoch}, Train Loss: {train_loss:.5f}, Train Accuracy: {train_accu:.4f}, Val Accuracy: {val_accu:.4f}")
            if exp_info is not None:
                wandb.log({
                    "train_loss": train_loss, 
                    "train_accu": train_accu, 
                    "val_accu": val_accu, 
                    "epoch": epoch,
                    "val_loss": val_loss,
                })
            if val_accu > best_val_accu:
                exit_epoch = epoch
                best_val_accu = val_accu
                best_model_state = model.state_dict()

    # 加载最佳模型并在测试集上评估
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for test_X, test_y in test_loader:
            test_X = test_X.to(DEVICE)
            test_y = test_y.to(DEVICE)
            test_outputs = model(test_X)
            test_preds = test_outputs.argmax(dim=1)
            test_correct += (test_preds == test_y).sum().item()
            test_total += test_X.size(0)
    test_accu = test_correct / test_total
    print(f"Exit Epoch: {exit_epoch}")
    print(f"Addition Accuracy: {test_accu}")
    wandb.log({"test_accu": test_accu, "orderness": exp_info['orderness'], 'layer': exp_info['layer']})
    if exp_info is not None:
        wandb.finish()
    return test_accu



# if __name__ == "__main__":
#     input_dim = 768
#     output_dim = 10
#     linear = make_multi_layers(input_dim, output_dim, HIDDEN_PARAM, is_batch_norm=False, is_layer_norm=False)
#     print(linear)
