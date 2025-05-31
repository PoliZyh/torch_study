# https://www.kaggle.com/competitions/playground-series-s3e3/data

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader




FEATURES_MAP = {
    'BusinessTravel': {
        'Travel_Rarely': 0,
        'Travel_Frequently': 1,
        'Non-Travel': 2
    },
    'Department': {
        'Research & Development': 0,
        'Sales': 1,
        'Human Resources': 2
    },
    'Gender': {
        'Male': 0,
        'Female': 1
    },
    'JobRole': {
        'Sales Executive': 0,
        'Research Scientist': 1,
        'Laboratory Technician': 2,
        'Manufacturing Director': 3,
        'Healthcare Representative': 4
    },
    'MaritalStatus': {
        'Married': 0,
        'Single': 1,
        'Divorced': 2
    },
    'OverTime': {
        'Yes': 0,
        'No': 1
    }
}
COLUMNS_TO_SCALE = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
                   'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike',
                   'TotalWorkingYears', 'YearsAtCompany']
LABEL_TITLE = 'Attrition'
BATCH_SIZE = 64
DATASET_TRAIN_FILEPATH = '../../dataset/tabular_employee_attrition/train.csv'
DATASET_TEST_FILEPATH = '../../dataset/tabular_employee_attrition/test.csv'
OUTPUTS_FILEPATH = '../../dataset/tabular_employee_attrition/submission.csv'

class AttritionDataset(Dataset):
    def __init__(self, filepath, is_train):
        df = pd.read_csv(filepath)

        df = df.drop(columns=['id', 'EducationField', 'EmployeeCount', 'Over18', 'StandardHours'])
        df['BusinessTravel'] = df['BusinessTravel'].map(FEATURES_MAP['BusinessTravel'])
        df['Department'] = df['Department'].map(FEATURES_MAP['Department'])
        df['Gender'] = df['Gender'].map(FEATURES_MAP['Gender'])
        df['JobRole'] = df['JobRole'].map(FEATURES_MAP['JobRole'])
        df['MaritalStatus'] = df['MaritalStatus'].map(FEATURES_MAP['MaritalStatus'])
        df['OverTime'] = df['OverTime'].map(FEATURES_MAP['OverTime'])

        for col in COLUMNS_TO_SCALE:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

        # 检查是否有NaN值
        if df.isnull().any().any():
            print("Warning: NaN values found in data!")
            df = df.fillna(0)  # 填充NaN值为0

        if is_train and LABEL_TITLE in df.columns:
            self.y_data = torch.tensor(df[LABEL_TITLE].values).float() # 应该改为（BCELoss需要float）
            self.x_data = torch.tensor(df.drop(columns=[LABEL_TITLE]).values).float()
        else:
            self.x_data = torch.tensor(df.values).float()
            self.y_data = None

        self.len = len(self.x_data)

    def __getitem__(self, idx):
        if self.y_data is not None:
            return self.x_data[idx], self.y_data[idx]
        return self.x_data[idx]

    def __len__(self):
        return self.len


train_dataset = AttritionDataset(DATASET_TRAIN_FILEPATH, True)
test_dataset = AttritionDataset(DATASET_TEST_FILEPATH, False)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(29, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred


model = Net()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data

        # 修改这里：将labels reshape为 [batch_size, 1]
        labels = labels.float().unsqueeze(1)  # [64] -> [64, 1]

        y_pred = model(inputs)
        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 20 == 19:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 20))
            running_loss = 0.0

def test():
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            inputs = data
            outputs = model(inputs)
            preds = outputs.flatten().detach().cpu().tolist()
            all_preds.extend(preds)

    return all_preds


if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)

        # 测试模型并生成预测
    predictions = test()  # 获取所有batch的预测概率
    print(predictions)

    test_df = pd.read_csv(DATASET_TEST_FILEPATH)

    submission = pd.DataFrame({
        'id': test_df['id'].values,
        'Attrition': predictions
    })

    submission.to_csv(OUTPUTS_FILEPATH, index=False)
    print(f"Submission file saved to {OUTPUTS_FILEPATH}")