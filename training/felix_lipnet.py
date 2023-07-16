from dataset import FelixLRS2Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn



class FelixLipNet(nn.Module):
    def __init__(self, num_classes):
        super(FelixLipNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 128, kernel_size=3, padding=1) 
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv3 = nn.Conv3d(256, 300, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm1 = nn.LSTM(43200, 128, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x)) # N, 1, 300, 96, 96 -> N, 128, 300, 96, 96
        x = self.maxpool1(x) # N, 128, 300, 96, 96 -> N, 128, 300, 48, 48
        x = self.relu(self.conv2(x)) # N, 128, 300, 48, 48 -> N, 256, 300, 48, 48
        x = self.maxpool2(x) # N, 256, 300, 48, 48 -> N, 256, 300, 24, 24
        x = self.relu(self.conv3(x)) # N, 256, 300, 24, 24 -> N, 300, 300, 24, 24
        x = self.maxpool3(x) # N, 256, 300, 24, 24 -> N, 256, 300, 12, 12
        x = self.flatten(x) # N, 256, 300, 12, 12 -> N, 300, 43200
        
        x, _ = self.lstm1(x) # N, 300, 43200 -> N, 300, 256
        x = self.dropout1(x)
        x, _ = self.lstm2(x) # N, 300, 256 -> N, 300, 256
        x = self.dropout2(x)
        x = self.fc(x) # N, 300, 256 -> N, 300, 41
        x = self.log_softmax(x)
        return x



# Create our vocab list
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!0123456789 "]
char_to_num = LabelEncoder()
char_to_num.fit(vocab)

# Create an instance of the PyTorch model
num_classes = len(char_to_num.classes_) + 1
model = FelixLipNet(num_classes)