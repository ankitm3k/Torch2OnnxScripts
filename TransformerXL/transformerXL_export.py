import torch
import transformers
from transformers import TransfoXLTokenizer, TransfoXLModel

tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
model = TransfoXLModel.from_pretrained("transfo-xl-wt103")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

torch.onnx.export(
    model,
    tuple(inputs.values()),
    f="txl.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'},
                  'attention_mask': {0: 'batch_size', 1: 'sequence'},
                  'logits': {0: 'batch_size', 1: 'sequence'}},
    do_constant_folding=True,
    opset_version=14,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
)

'''
SUCCESS -

operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK

FAIL -
operator_export_type = default

torch.onnx.symbolic_registry.UnsupportedOperatorError: Exporting the operator ::triu to ONNX opset version 13 is not supported. 
Support for this operator was added in version 14, try exporting with this version.

if used opset 14 - RuntimeError: Expected node type 'onnx::Constant' for argument 'diagonal' of node 'triu', got 'onnx::Add'.
'''
