import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
class_mapping = {
    'Class_1': 0,  # 通常从0开始
    'Class_2': 1,
    'Class_3': 2,
    'Class_4': 3,
    'Class_5': 4,
    'Class_6': 5,
    'Class_7': 6,
    'Class_8': 7,
    'Class_9': 8   # 最后一个类别是8（如果是0-based）
}
dataset_train_filepath = '../dataset/otto/train.csv'
dataset_test_filepath = '../dataset/otto/test.csv'
outputs_filepath = '../dataset/otto/submission.csv'

class OttoDataset(Dataset):
    def __init__(self, filepath, is_train):
        df = pd.read_csv(filepath)

        if is_train and 'target' in df.columns:
            df['target'] = df['target'].map(class_mapping)
            # 先转换为 NumPy 数组，再转为 Tensor
            self.x_data = torch.tensor(df.drop(columns=['id', 'target']).values).float()
            self.y_data = torch.tensor(df['target'].values).long()
        else:
            self.x_data = torch.tensor(df.drop(columns=['id']).values).float()
            self.y_data = None

        self.len = len(self.x_data)

    def __getitem__(self, idx):
        if self.y_data is not None:
            return self.x_data[idx], self.y_data[idx]
        return self.x_data[idx]

    def __len__(self):
        return self.len


train_dataset = OttoDataset(dataset_train_filepath, True)
test_dataset = OttoDataset(dataset_test_filepath, False)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(93, 80)
        self.l2 = torch.nn.Linear(80, 64)
        self.l3 = torch.nn.Linear(64, 32)
        self.l4 = torch.nn.Linear(32, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            inputs = data
            outputs = model(inputs)
            # _, predicted = torch.max(outputs.data, dim=1)
            probs = F.softmax(outputs, dim=1)  # 转换为概率
            all_preds.append(probs.cpu().numpy())
            # all_preds.extend(predicted.cpu().numpy())

    return all_preds

if __name__ == '__main__':

    for epoch in range(20):  # 训练10个epoch
        train(epoch)

        # 测试模型并生成预测
    predictions = test()  # 获取所有batch的预测概率

    # 合并所有batch的预测结果
    predictions = np.concatenate(predictions, axis=0)  # shape: (n_samples, 9)

    # 读取测试集原始文件获取id
    test_df = pd.read_csv(dataset_test_filepath)

    # 创建提交DataFrame
    submission = pd.DataFrame(
        predictions,
        columns=[f'Class_{i}' for i in range(1, 10)]  # 列名: Class_1到Class_9
    )

    # 添加id列到第一列
    submission.insert(0, 'id', test_df['id'].values)

    # 保存为CSV文件
    submission.to_csv(outputs_filepath, index=False)
    print(f"Submission file saved to {outputs_filepath}")