import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class TitanicDataset(Dataset):
    def __init__(self, filepath, train):
        # auto skip the row of title
        df = pd.read_csv(filepath)
        # recode
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(np.float32)
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
        df['Fare'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()
        self.x_data = df[['Pclass', 'Sex', 'Age', 'Fare']].values.astype(np.float32)
        self.y_data = None
        if train and 'Survived' in df.columns:
            self.y_data = df[['Survived']].values.astype(np.float32)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        if self.y_data is not None:
            return self.x_data[index], self.y_data[index]
        return self.x_data[index]  # 测试集仅返回特征

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred


train_dataset = TitanicDataset('../dataset/titanic/train.csv', True)
test_dataset = TitanicDataset('../dataset/titanic/test.csv', False)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32,
                         shuffle=False,
                         num_workers=2)

model = Model()
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def save_test_predictions(model, test_loader, test_file_path, output_file):
    """
    保存测试集预测结果为CSV文件（格式：PassengerId, Survived）

    参数：
        model: 训练好的模型
        test_loader: 测试集DataLoader
        test_file_path: 原始测试集CSV路径（用于获取PassengerId）
        output_file: 输出文件名
    """
    model.eval()
    all_preds = []

    # 1. 获取预测结果
    with torch.no_grad():
        for data in test_loader:
            inputs = data  # 测试集只有特征
            outputs = model(inputs)
            preds = (outputs >= 0.5).int().flatten().tolist()  # 概率转0/1
            all_preds.extend(preds)

    # 2. 读取原始测试文件获取PassengerId
    test_df = pd.read_csv(test_file_path)
    passenger_ids = test_df['PassengerId'].values

    # 3. 确保数量匹配（必须418行）
    assert len(passenger_ids) == 418, f"Expected 418 test samples, got {len(passenger_ids)}"
    assert len(all_preds) == 418, f"Expected 418 predictions, got {len(all_preds)}"

    # 4. 生成提交文件
    submission_df = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': all_preds
    })

    submission_df.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")
    return submission_df




if __name__ == "__main__":
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # Prepare data
            inputs, labels = data
            # Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print("epoch =", epoch, "i =", i, "loss =", loss.item())
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()

    test_file_path = '../dataset/titanic/test.csv'

    # 生成提交文件
    save_test_predictions(
        model=model,
        test_loader=test_loader,
        test_file_path=test_file_path,
        output_file="../dataset/titanic/submission.csv"
    )