from qlib.model.base import Model
from typing import Text, Union
from qlib.data.dataset import Dataset
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from qlib.data.dataset.handler import DataHandlerLP


#使用qlib的数据集
import torch.nn as nn
from torchinfo import summary
import torch

#模型构建
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """

        :param inplanes: 输入通道数
        :param planes:  输出通道数
        :param stride: 步长
        :param downsample:  基础结构里有一个从x直接连到下面的线，
        如果上一个ResidualBlock的输出维度和当前的ResidualBlock的维度不一样，
        那就对这个x进行downSample操作，如果维度一样，直接加就行了，这时直接output=x+residual
        """

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = x + residual
        out = self.relu(out)

        return out


class ResNet_18(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        """

        :param block:  调用basicblock
        :param layers:  使用一个列表 储存想搭建的layer
        :param num_classes: fc的输出维度
        """

        self.inplanes = 64
        super(ResNet_18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 构建layer
    def make_layer(self, block, planes, blocks, stride=1):
        """

        :param block: 调用basicblock
        :param planes: 输出通道大小
        :param blocks: 调用列表
        :param stride: 步长
        :return: nn.Sequential(*layers)
        """
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output

def fit_model(self, x_train, y_train, x_valid, y_valid):
    model = ResNet_18(BasicBlock,[2,2,2,2])
    model.compile(optimizer='adam',
                  loss=self.loss,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=self.epochs, validation_data=(x_valid, y_valid), verbose=2)
    model.summary()
    model.save('model')
def fit(self, dataset: Dataset):

    self.loss =torch.ctc_loss()
    df_train, df_valid, df_test = dataset.prepare(
        ["train", "valid", "test"],
        col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
    )
    x_train, y_train = df_train["feature"], df_train["label"]
    x_valid, y_valid = df_valid["feature"], df_valid["label"]
    self.fit_model(x_train, y_train, x_valid, y_valid)
    #预测函数
def predict(self, dataset: Dataset, segment: Union[Text, slice] = "test"):
    test = dataset.prepare(segment,
                           col_set="feature",
                           data_key=DataHandlerLP.DK_I)

    model_trained = torch.load('model')
    #加载之前保存的model
    model = tf.keras.Sequential([model_trained,
                                             tf.keras.layers.Softmax()])

    predict = model.predict(test)
    #输出
    print(predict)

    return predict


