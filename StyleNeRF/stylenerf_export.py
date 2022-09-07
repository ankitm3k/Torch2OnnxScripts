import torch
import dnnlib
import imageio
import legacy
import os
import glob
from training.networks import Generator
from torch_utils import misc

'''
Pre-requisites before converting to onnx, we need to perform few changes in below files -
1) conv2d_gradfix.py
   change input and weights in line #37
   def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    input = input.to(torch.float32)
    weight = weight.to(torch.float32)
    ... ...
2) networks.py
   comment line #855 - #assert x.dtype == dtype
   
These changes are needed if the user wants to export a model purely for torch.device('cpu') 
else one can get error wrt conv2d operator as -
RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
as this conv2d op implementation is not available for cpu
'''

network_pkl = 'ffhq_512.pkl'

device = torch.device('cpu')
if os.path.isdir(network_pkl):
    network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
print('Loading networks from "%s"...' % network_pkl)

with dnnlib.util.open_url(network_pkl) as f:
    network = legacy.load_network_pkl(f)
    G = network['G_ema'].to(device) # type: ignore
    D = network['D'].to(device)

with torch.no_grad():
    G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
    misc.copy_params_and_buffers(G, G2, require_all=False)
    G2 = G2.to(torch.float32)
    G2.eval()
    input = torch.zeros([1,512])
    # Exporting generator only
    torch.onnx.export(G2, input,"stylenerf.onnx",verbose=True,opset_version=14,
                      operator_export_type= torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)