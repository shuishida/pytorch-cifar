from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .dla_simple import *
from .dla import *


models = {
    "vgg19": VGG('VGG19'),
    "resnet18": ResNet18(),
    "preact18": PreActResNet18(),
    "googlenet": GoogLeNet(),
    "densenet121": DenseNet121(),
    "resnext29_2": ResNeXt29_2x64d(),
    "mobilenet": MobileNet(),
    "mobilenetv2": MobileNetV2(),
    "dpn92": DPN92(),
    # "shufflenetg2": ShuffleNetG2(),
    "senet18": SENet18(),
    "shufflenetv2": ShuffleNetV2(1),
    "efficientnetb0": EfficientNetB0(),
    "regnetx_200": RegNetX_200MF(),
    "simpledla": SimpleDLA(),
}
