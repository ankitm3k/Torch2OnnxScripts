import torch
from ssd.ssd300 import SSD300

def ssd300():
    num_classes = 21
    net = SSD300(num_classes)
    net.eval()
    dummy_input = torch.zeros([1, 3, 512, 512])
    torch.onnx.export(net,dummy_input,"ssd300.onnx",verbose=True)

if __name__ == '__main__':
    ssd300()