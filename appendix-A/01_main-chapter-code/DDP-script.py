# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Appendix A: Introduction to PyTorch (Part 3)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# NEW imports:
import os
import torch.multiprocessing as mp # 多进程库
from torch.utils.data.distributed import DistributedSampler # 分布式采样器 
from torch.nn.parallel import DistributedDataParallel as DDP # 分布式数据并行包装器
from torch.distributed import init_process_group, destroy_process_group # 用于初始化和销毁进程组的函数

"""
这段代码是分布式训练的初始化部分，它设置了进程组和通信机制，使得每个 GPU 上的进程可以协同工作。
在实际的训练代码中，你还需要设置分布式采样器 DistributedSampler，确保每个进程只处理数据集的一部分，
以及使用 DistributedDataParallel 来包装你的模型，使得模型的参数在所有进程中同步更新
"""
# NEW: function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
# 初始化分布式进程组。每个 GPU 对应一个进程。
# 这个函数接受两个参数：rank（当前进程的唯一 ID）和 world_size（进程组中的总进程数）
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # rank of machine running rank:0 process
    # here, we assume all GPUs are on the same machine
    os.environ["MASTER_ADDR"] = "localhost" #  rank 0 进程的机器的地址 这里假设所有的 GPU 都在同一台机器上，所以地址设置为 localhost
    # any free port on the machine
    os.environ["MASTER_PORT"] = "12345" # 进程组通信所用的端口号。这里随便选了一个端口号 12345，实际上应该选择一个当前未被占用的端口

    # initialize process group
    # Windows users may have to use "gloo" instead of "nccl" as backend
    # nccl: NVIDIA Collective Communication Library
    # init_process_group 函数用于设置进程组，以便进程之间可以通信。
    # 这里使用的后端是 nccl，它是 NVIDIA 的一个库，专门用于在 NVIDIA GPU 上进行高效的集体通信
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: False because of DistributedSampler below
        pin_memory=True,
        drop_last=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        # 用于确保在多个 GPU 上进行训练时，每个 GPU 处理的数据批次（batch）不会重叠，即每个样本只被一个 GPU 处理
        sampler=DistributedSampler(train_ds)  # NEW
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# NEW: wrapper
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # NEW: initialize process groups

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # rank 是当前进程的唯一 ID，用于确定当前进程应该使用哪个 GPU。在分布式训练中，每个进程通常只使用一个 GPU
    model = DDP(model, device_ids=[rank])  # NEW: wrap model with DDP
    # the core model is now accessible as model.module

    for epoch in range(num_epochs):

        model.train()
        for features, labels in train_loader:

            # rank 是当前进程的唯一 ID，用于确定当前进程应该使用哪个 GPU。在分布式训练中，每个进程通常只使用一个 GPU
            features, labels = features.to(rank), labels.to(rank)  # New: use rank
            logits = model(features)
            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)

    destroy_process_group()  # NEW: cleanly exit distributed mode


def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())

    torch.manual_seed(123)

    # NEW: spawn new processes
    # note that spawn will automatically pass the rank
    num_epochs = 3
    # 获取了机器上可用的 GPU 数量，这将作为进程组的总大小
    world_size = torch.cuda.device_count()
    
    # spawn 函数来启动多个进程。spawn 会为每个 GPU 启动一个进程，每个进程都会调用 main 函数
    # nprocs=world_size 指定了要启动的进程数量，这里设置为 GPU 的数量，意味着每个 GPU 对应一个进程
    """
    在 Python 的 multiprocessing 模块中，spawn 函数用于创建新的进程。
    当你使用 spawn 启动新的进程时，你可以指定一个函数（在这个例子中是 main 函数）以及传递给这个函数的参数。

    这里的 args 参数是一个元组，包含了所有要传递给 main 函数的参数。
    spawn 会自动将这个元组解包，并将元组中的每个元素作为单独的参数传递给 main 函数。

    rank 是由 spawn 自动传递的当前进程的排名:
    torch.multiprocessing.spawn 来启动多个进程时，rank 参数是自动传递给 main 函数的。
    rank 是一个标识符，用于区分不同的进程，它代表当前进程在所有进程中的序号.
    当你调用 spawn 函数时，它会为每个进程创建一个新的 Python 解释器，并在这个新的解释器中导入你的脚本。
    在这个过程中，spawn 函数会自动设置一个环境变量 RANK（在某些情况下也可能是 LOCAL_RANK），并将当前进程的 rank 值存储在这个环境变量中。
    然后，在你的 main 函数中，你可以读取这个环境变量来获取当前进程的 rank

    调用 mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size) 时，
    实际上是在为每个进程创建一个参数列表 [world_size, num_epochs]，并将这个列表传递给 main 函数。
    每个进程都会收到自己的 rank 和相同的 world_size 和 num_epochs 参数
    """
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size spawns one process per GPU
