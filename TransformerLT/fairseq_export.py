import torch
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
mname = "facebook/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

input = "Machine learning is great, isn't it?"
input_ids = tokenizer.encode(input, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(decoded) # expected out - Maschinelles Lernen ist großartig, oder?


# https://huggingface.co/facebook/wmt19-en-de

torch.onnx.export(model,input_ids,"fairseq.onnx",verbose=True)
