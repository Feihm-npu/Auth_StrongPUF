import torch
import torch.nn as nn

class VVG16_net(nn.Module):
    def __init__(self, in_channels=3, n_classes=1000, img_size=(224, 224)):  # 标准VGG16为RGB图像，默认输入通道为3
        super().__init__()
        self.in_channels = in_channels
        self.act = nn.ReLU()
        self.conv1 = self.conv_block(in_channels=self.in_channels, block=[64, 64])
        self.conv2 = self.conv_block(in_channels=64, block=[128, 128])
        self.conv3 = self.conv_block(in_channels=128, block=[256, 256, 256])
        self.conv4 = self.conv_block(in_channels=256, block=[512, 512, 512])
        self.conv5 = self.conv_block(in_channels=512, block=[512, 512, 512])
        
        # 计算卷积输出的特征图尺寸
        conv_output_size = self._get_conv_output(img_size)
        print(f'conv_output_size: {conv_output_size}')
        
        # 标准VGG16的全连接层
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 4096),  # VGG16有两个4096的全连接层
            self.act,
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),  # 第二个全连接层也是4096维
            self.act,
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes)  # 最后是输出层，n_classes个类别
        )

    def _get_conv_output(self, img_size):
        # 创建一个虚拟输入图像，用于计算卷积层输出的特征图尺寸
        dummy_input = torch.zeros(1, self.in_channels, *img_size)
        dummy_output = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
        return int(torch.prod(torch.tensor(dummy_output.size()[1:])))  # 计算输出尺寸

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fcs(x)
        return x

    def conv_block(self, in_channels ,block):
        layers = []
        for i in block:
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=(2,2), stride=(1,1), padding=(1,1)),
                       nn.BatchNorm2d(i),  # 扩展：VGG16原版没有BatchNorm层，添加此层有助于训练
                       self.act]
            in_channels = i
        layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]  # 池化层：VGG16使用最大池化层
        return nn.Sequential(*layers)

