import torch
import torch.nn as nn

assert __name__ != '__main__', 'Module startup error.'


def weights_init(model):
    class_name = model.__class__.__name__

    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(128 + 52, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 6272),
            nn.BatchNorm1d(6272),
            nn.ReLU())

        self.transpose1 = nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, (6, 6)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.transpose2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.transpose3 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, (3, 3)),
            nn.Tanh())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, y), dim=1)
        x = self.fc1(x)
        x = self.fc2(x).view(-1, 128, 7, 7)

        for transpose in (self.transpose1, self.transpose2, self.transpose3):
            x = transpose(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (5, 5), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))

        self.fc1 = nn.Sequential(
            nn.Linear(2048 + 52, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2))

        self.fc2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(torch.cat((x.view(-1, 2048), y), dim=1))
        x = self.fc2(x)

        return x


# class Decorator(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, (5, 5)),
#             nn.BatchNorm2d(32),
#             nn.ReLU())
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 1, (3, 3)),
#             nn.Tanh())
#
#         nn.init.kaiming_normal_(self.conv1.weight)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         pass


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, (3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, (5, 5), stride=(2, 2)),
            nn.Dropout2d(0.4),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 64, (5, 5), stride=(2, 2)),
            nn.Dropout2d(0.4),
            nn.ReLU())

        self.fc1 = nn.Linear(256, 52)

        for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            nn.init.kaiming_normal_(layer[0].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            x = conv(x)

        x = x.view(-1, 256)
        x = self.fc1(x)
        return x
