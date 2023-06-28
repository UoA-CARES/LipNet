from dataset import FelixLRS2Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn



class FelixLipNet(nn.Module):
    def __init__(self, num_classes):
        super(FelixLipNet, self).__init__()
        self.conv1 = nn.Conv3d(300, 128, kernel_size=3, padding=1) 
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.conv3 = nn.Conv3d(256, 10000, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm1 = nn.LSTM(144, 128, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x)) # 2, 300, 96, 96, 1 -> 2, 128, 96, 96, 1
        x = self.maxpool1(x) # 2, 128, 96, 96, 1 -> 2, 128, 48, 48, 1
        x = self.relu(self.conv2(x)) # 2, 128, 48, 48, 1 -> 2, 256, 48, 48, 1
        x = self.maxpool2(x) # 2, 256, 48, 48, 1 -> 2, 256, 24, 24, 1
        x = self.relu(self.conv3(x)) # 2, 256, 24, 24, 1 -> 2, 10000, 24, 24, 1
        x = self.maxpool3(x) # 2, 10000, 24, 24, 1 -> 2, 75, 12, 12, 1
        x = self.flatten(x) # 2, 10000, 12, 12, 1 -> 2, 10000, 144

        x, _ = self.lstm1(x) # 2, 10000, 144 -> 2, 10000, 256
        x = self.dropout1(x)
        x, _ = self.lstm2(x) # 2, 10000, 256 -> 2, 10000, 256
        x = self.dropout2(x)
        x = self.fc(x) # 2, 10000, 256 -> 2, 10000, 40
        x = self.softmax(x)
        return x



# Create our vocab list
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "] + ['']
char_to_num = LabelEncoder()
char_to_num.fit(vocab)

# Create an instance of the PyTorch model
num_classes = len(char_to_num.classes_)
model = FelixLipNet(num_classes)