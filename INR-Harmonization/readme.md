# INR-Harmonization  Inference & ONNX Model Export
Clone desired Github Repo. 

>`git clone https://github.com/WindVChen/INR-Harmonization.git` \
>`cd INR-Harmonization`

Paste the `harmonization_inference_and_onnx_export.py` file in the root of the above repo directory i.e. your-desired-path/INR-Harmonization.

1. Install dependencies in your virtual environment.
> `pip install -r requirements.txt`

2. Download the pretrained weights for HRNetV2_Imagenet architecture.
- [HRNetV2_Imagenet](https://onedrive.live.com/?authkey=%21AMkPimlmClRvmpw&id=F7FD0B7F26543CEB%21112&cid=F7FD0B7F26543CEB&parId=root&parQt=sharedby&parCid=UnAuth&o=OneUp)

3. Modify below files with suggested modifications -
- `INR-Harmonization\model\backbone.py`\
    Correct path in line #69 with `self.load_pretrained_weights(r"./pretrained_models/hrnetv2_w18_imagenet_pretrained.pth")`
- `INR-Harmonization\model\base\conv_autoencoder.py`\
    Change line #245 with `coord = misc.get_mgrid(self.opt.INR_input_size // (2 ** (self.max_hidden_mlp_num - n - 1))).unsqueeze(0).repeat(encoder_outputs[0].shape[0], 1, 1).to(self.opt.device)` \
    Change line #259 with `res_h = res_w = image.shape[-2] // (2 ** (self.max_hidden_mlp_num - n - 1))`
- `INR-Harmonization\utils\misc.py`\
    Change line #96 with `width = int(np.sqrt(num_samples))`

4. Run the inference and ONNX export using below command


    > `python harmonization_inference_and_onnx_export.py --split_resolution 256 --pretrained ./pretrained_models/hrnetv2_w18_imagenet_pretrained.pth`

Additional info -
- For a successful ONNX export, one need to register `Unfold` Op before calling the export function else it might lead to `RuntimeError: Unsupported: ONNX export of operator Unfold, input size not accessible. Please feel free to request support or submit a pull request on  PyTorch GitHub.`
- Export config should have `opset_version=16` and use `OperatorExportTypes.ONNX_ATEN_FALLBACK`.
- (Optional) Finally, one can use [ONNX Simplifier](https://pypi.org/project/onnx-simplifier/) to further simplify and optimize the ONNX model.