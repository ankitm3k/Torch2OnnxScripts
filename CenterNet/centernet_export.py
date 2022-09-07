import torch
import io
import argparse

from torch.onnx import OperatorExportTypes
from types import MethodType
from src.lib.models.model import create_model, load_model

def dlav0_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x[self.first_level:])
    ret = []
    for head in self.heads:
        ret.append(self.__getattr__(head)(x))
    return ret

# define args
parser = argparse.ArgumentParser()
parser.add_argument('--torch_model_path',
                    type=str,
                    help='Path to target torch model to be converted to ONNX')
args = parser.parse_args()
		
dummy_input = torch.zeros([1, 3, 512, 512])

model_arch = 'dlav0_34'
heads = {'hm': 10, 'wh': 2, 'reg': 2}
head_conv = 256
model = create_model(model_arch, heads, head_conv)
model.forward = MethodType(dlav0_forward, model)
model = load_model(model, args.torch_model_path)
model.eval()

torch.onnx.export(model, dummy_input, "centernet_multi_pose_dla_3x.onnx", verbose=False,
                  operator_export_type=OperatorExportTypes.ONNX)
