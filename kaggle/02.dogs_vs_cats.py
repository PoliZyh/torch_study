import torch
import os
from kaggle.utils.img import calculate_mean_std
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import pandas as pd

BATCH_SIZE = 64
IMG_SIZE = 224
DATA_TRAIN_PATH = '../../dataset/dogs_vs_cats/train/train'
DATA_TEST_PATH = '../../dataset/dogs_vs_cats/test/test'
OUTPUT_PATH = '../../dataset/dogs_vs_cats/submission.csv'
mean = [0.4883, 0.4551, 0.4170]
std = [0.2257, 0.2211, 0.2214]

transform = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class DogsCatsDataset(Dataset):
    def __init__(self, folder_path, is_train, transform):
        self.folder_path = folder_path
        self.is_train = is_train
        self.transform = transform
        self.filenames = [
            f for f in os.listdir(folder_path) if f.endswith('.jpg')
        ]
        self.labels = None
        if is_train:
            self.labels = [
                0 if 'cat' in fname.lower() else 1
                for fname in self.filenames
            ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.filenames[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if self.is_train:
            label = self.labels[idx]
            return image, label
        return image


train_dataset = DogsCatsDataset(DATA_TRAIN_PATH, True, transform)
test_dataset = DogsCatsDataset(DATA_TEST_PATH, False, transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(20 * 53 * 53, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return self.sigmoid(x)

model = Net()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        target = target.float().unsqueeze(1)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 10))
            running_loss = 0.0


def test():
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in test_loader:
            inputs = data
            outputs = model(inputs)
            all_preds.extend(outputs.squeeze(1).cpu().numpy().tolist())

    submission_df = pd.DataFrame({
        'id': range(1, len(all_preds) + 1),
        'label': all_preds
    })

    submission_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission file saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    for epoch in range(2):
        train(epoch)
    test()
