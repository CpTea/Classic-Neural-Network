import tensorflow as tf 
from tensorflow import keras


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(outplanes, strides=1):
    return keras.layers.Conv2D(
        outplanes,
        kernel_size=3,
        strides=strides,
        padding='SAME',
        use_bias=False)


class BasicBlock(keras.Model):
    expansion = 1

    def __init__(self, planes, strides=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(planes, strides)
        self.bn1 = keras.layers.BatchNormalization(axis=3)
        self.relu = keras.layers.ReLU()
        self.conv2 = conv3x3(planes)
        self.bn2 = keras.layers.BatchNormalization(axis=3)
        self.downsample = downsample
        self.strides = strides

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out = keras.layers.add([out, residual])
        out = self.relu(out)
 
        return out


class Bottleneck(keras.Model):
    expansion = 4

    def __init__(self, planes, strides=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = keras.layers.Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn1 = keras.layers.BatchNormalization(axis=3)
        self.conv2 = keras.layers.Conv2D(planes, kernel_size=3, strides=strides, padding='SAME', use_bias=False)
        self.bn2 = keras.layers.BatchNormalization(axis=3)
        self.conv3 = keras.layers.Conv2D(planes*4, kernel_size=1, use_bias=False)
        self.bn3 = keras.layers.BatchNormalization(axis=3)
        self.relu = keras.layers.ReLU()

        self.downsample = downsample
        self.strides = strides
    
    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = keras.layers.add([out, residual])
        out = self.relu(out)

        return out


class ResNet(keras.Model):
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='SAME', use_bias=False)
        self.bn1 = keras.layers.BatchNormalization(axis=3)
        self.relu = keras.layers.ReLU()
        self.maxpool = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], strides=2)
        self.layer3 = self._make_layer(block, 256, layers[2], strides=2)
        self.layer4 = self._make_layer(block, 512, layers[3], strides=2)
        self.avgpool = keras.layers.AveragePooling2D(pool_size=7, strides=1)
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(num_classes, activation=tf.keras.layers.Softmax())

    def _make_layer(self, block, planes, blocks, strides=1):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.models.Sequential([
                keras.layers.Conv2D(
                    planes * block.expansion, 
                    kernel_size=1, strides=strides,
                    use_bias=False),                
                keras.layers.BatchNormalization(axis=3)])
        
        layers = []
        layers.append(block(planes, strides, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(block(planes))

        return keras.models.Sequential(layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)