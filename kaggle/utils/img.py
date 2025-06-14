import torch


def calculate_mean_std(channel, dataloader):
    mean = torch.zeros(channel)
    std = torch.zeros(channel)
    total = 0

    print("开始计算mean, std...")
    for data, _ in dataloader:
        batch_size = data.size(0) # batch size
        data = data.view(batch_size, data.size(1), -1) # (N, C, H*W)
        mean += data.mean(2).sum(0) # 对每个通道求均值
        std += data.std(2).sum(0) # 对每个通道求标准差
        total += batch_size

    mean /= total
    std /= total

    return mean, std

