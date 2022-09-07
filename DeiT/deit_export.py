import torch

def deit():
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    model.eval()
    dummy_input = torch.zeros([1, 3, 224, 224])
    torch.onnx.export(model,dummy_input,"deit.onnx",verbose=True)

if __name__ == '__main__':
    deit()