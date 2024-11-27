import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class CNNModel(nn.Module):
    def __init__(self, no_of_classes):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.activation2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(512)
        self.activation3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.activation4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(4608, 256)
        self.batchnorm_fc1 = nn.BatchNorm1d(256)
        self.activation_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(256, 512)
        self.batchnorm_fc2 = nn.BatchNorm1d(512)
        self.activation_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(0.25)

        self.fc3 = nn.Linear(512, no_of_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout1(self.maxpool1(self.activation1(self.batchnorm1(self.conv1(x)))))
        x = self.dropout2(self.maxpool2(self.activation2(self.batchnorm2(self.conv2(x)))))
        x = self.dropout3(self.maxpool3(self.activation3(self.batchnorm3(self.conv3(x)))))
        x = self.dropout4(self.maxpool4(self.activation4(self.batchnorm4(self.conv4(x)))))

        # Flatten the output from the last convolutional layer
        x = x.view(x.size(0), -1)

        x = self.dropout_fc1(self.activation_fc1(self.batchnorm_fc1(self.fc1(x))))
        x = self.dropout_fc2(self.activation_fc2(self.batchnorm_fc2(self.fc2(x))))
        
        x = self.fc3(x)

        return x

# Load the model
model = CNNModel(no_of_classes=7)  # Assuming 7 emotion classes
model.load_state_dict(torch.load('CNN_model.pth'))
model.eval()