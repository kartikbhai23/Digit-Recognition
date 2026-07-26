import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        conv1_filters=64,
        conv2_filters=32,
        fc_units=256,
        dropout_rate=0.3,
        num_classes=10,
    ):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv1_filters,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=conv1_filters,
                out_channels=conv2_filters,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=conv2_filters * 5 * 5,
                out_features=fc_units
            ),

            nn.ReLU(),

            nn.Dropout(dropout_rate),

            nn.Linear(
                fc_units,
                num_classes
            ),
        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x