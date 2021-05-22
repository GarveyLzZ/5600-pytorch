import torch
import torch.nn as nn
import torchvision

print("Pytorch version", torch.__version__)
print("Torchvision Version", torchvision.__version__)


# 卷积之后，如果要加入BatchNormalization操作，那个bias=False是有必要的，因为在BN操作时，占内存，还不起作用。
def Conv1(in_channels, out_channels, stride=2):
    model = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 7), stride=stride, padding=(0, 3),
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.Sigmoid(),
        nn.MaxPool2d((1, 2), stride=2, padding=(0,1))
    )
    return model


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), stride=stride,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=(1, 1),
                      stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        if self.downsampling:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.expansion, kernel_size=(1, 1),
                          stride=1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residule = x
        out = self.bottleneck(x)

        if self.downsampling:
            residule = self.downsampling(x)

        out += residule
        out = self.sigmoid(out)

        return out


class ResNet(nn.Module):
    def __init__(self, blocks=[3, 4, 6, 3], num_classes=1, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_channels=1, out_channels=64)

        self.layer1 = self.make_layers(in_channels=64, out_channels=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layers(in_channels=256, out_channels=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layers(in_channels=512, out_channels=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layers(in_channels=1024, out_channels=512, block=blocks[3], stride=2)

        self.avgpooling = nn.AvgPool2d((1, 7), stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='sigmoid')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, in_channels, out_channels, block, stride):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(out_channels * self.expansion, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# model = ResNet()
# print(model)
