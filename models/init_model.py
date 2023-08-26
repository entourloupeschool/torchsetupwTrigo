import torch.nn as nn
import torch
import torch.nn.functional as F

# create simple perceptron model


class Model(nn.Module):
    def __init__(self, num_classes=1, in_channels=2, f1_width=2, f2_width=3, f3_width=3, f4_width=3, f5_width=3, do_rate=0.2):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_channels, f1_width)
        self.bn1 = nn.BatchNorm1d(f1_width)
        self.fc2 = nn.Linear(f1_width, f2_width)
        self.bn2 = nn.BatchNorm1d(f2_width)
        self.fc3 = nn.Linear(f2_width, f3_width)
        self.bn3 = nn.BatchNorm1d(f3_width)
        self.fc4 = nn.Linear(f3_width, f4_width)
        self.bn4 = nn.BatchNorm1d(f4_width)
        self.fc5 = nn.Linear(f4_width, f5_width)
        self.bn5 = nn.BatchNorm1d(f5_width)
        self.fc6 = nn.Linear(f5_width, num_classes)

        # activation function
        self.relu = nn.ReLU()

        # utility layers
        self.do_rate = do_rate
        self.do = nn.Dropout1d(do_rate)

        # Metrics utiliy
        self.model_metrics = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.bn1(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.bn2(x)
        if self.do_rate > 0:
            x = self.do(x)
        x = self.fc3(x)
        x = self.relu(x)
        # x = self.bn3(x)
        if self.do_rate > 0:
            x = self.do(x)
        x = self.fc4(x)
        x = self.relu(x)
        # x = self.bn4(x)
        if self.do_rate > 0:
            x = self.do(x)
        x = self.fc5(x)
        x = self.relu(x)
        # x = self.bn5(x)
        # x = self.do(x)
        x = self.fc6(x)

        return x

    def save(self, model, name, path, EPOCH, optimizer):
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path+name+'.pth')

    def add_epoch_metrics(self, epoch_metrics):
        self.model_metrics.append(epoch_metrics)

    def get_model_metrics(self):
        return self.model_metrics


class CustomModel(nn.Module):
    def __init__(self, in_channels, layers_config, out_channels=1):
        super(CustomModel, self).__init__()

        self.layers = nn.ModuleList()
        prev_channels = in_channels

        for config in layers_config:
            neurons, batch_norm, dropout = config

            # Add linear layer
            self.layers.append(nn.Linear(prev_channels, neurons))

            # Add batch normalization layer if specified
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(neurons))

            # Add dropout layer if specified
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))

            prev_channels = neurons

        # Output layer
        self.out_layer = nn.Linear(prev_channels, out_channels)

        # Metrics utiliy
        self.model_metrics = []

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        x = self.out_layer(x)
        return x

    def save(self, model, name, path, EPOCH, optimizer):
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path+name+'.pth')

    def add_epoch_metrics(self, epoch_metrics):
        self.model_metrics.append(epoch_metrics)

    def get_model_metrics(self):
        return self.model_metrics
