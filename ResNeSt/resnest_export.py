import torch
from resnest.torch import resnest50

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import *

from resnest.gluon import get_model

def ResNeSt50():
    model = resnest50(pretrained=True)
    model.eval()
    dummy_input = torch.zeros([1, 3, 512, 512])
    torch.onnx.export(model,dummy_input,"{}.onnx".format(resnest50.__name__),verbose=False)



if __name__ == "__main__":
    ResNeSt50()
