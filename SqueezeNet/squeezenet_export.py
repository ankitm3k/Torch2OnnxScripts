import torch
from torchvision.models import squeezenet1_1

def squeezenetExport():
    net = squeezenet1_1()
    net.eval()
    dummy_input = torch.zeros([1, 3, 224, 224])
    torch.onnx.export(net,dummy_input,"squeezenetv11.onnx",verbose=True)

if __name__ == '__main__':
    squeezenetExport()