
import torch
from reformer_pytorch import ReformerLM, Reformer, LSHSelfAttention

def reformerLM():
    model = ReformerLM(
    num_tokens= 20000,
    dim = 1024,
    depth = 12,
    max_seq_len = 8192,
    heads = 8,
    lsh_dropout = 0.1,
    ff_dropout = 0.1,
    post_attn_dropout = 0.1,
    layer_dropout = 0.1,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
    causal = True,        # auto-regressive or not
    bucket_size = 64,     # average size of qk per bucket, 64 was recommended in paper
    n_hashes = 4,         # 4 is permissible per author, 8 is the best but slower
    emb_dim = 128,        # embedding factorization for further memory savings
    dim_head = 64,        # be able to fix the dimension of each head, making it independent of the embedding dimension and the number of heads
    ff_chunks = 200,      # number of chunks for feedforward layer, make higher if there are memory issues
    attn_chunks = 8,      # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
    num_mem_kv = 128,       # persistent learned memory key values, from all-attention paper
    full_attn_thres = 1024, # use full attention if context length is less than set value
    reverse_thres = 1024,   # turn off reversibility for 2x speed for sequence lengths shorter or equal to the designated value
    use_scale_norm = False,  # use scale norm from 'Transformers without tears' paper
    use_rezero = False,      # remove normalization and use rezero from 'ReZero is All You Need'
    one_value_head = False,  # use one set of values for all heads from 'One Write-Head Is All You Need'
    weight_tie = False,           # tie parameters of each layer for no memory per additional depth
    weight_tie_embedding = False, # use token embedding for projection of output, some papers report better results
    n_local_attn_heads = 2,       # many papers suggest mixing local attention heads aids specialization and improves on certain tasks
    pkm_layers = (4,7),           # specify layers to use product key memory. paper shows 1 or 2 modules near the middle of the transformer is best
    pkm_num_keys = 128,           # defaults to 128, but can be increased to 256 or 512 as memory allows
    use_full_attn = False    # only turn on this flag to override and turn on full attention for all sequence lengths. for comparison with LSH to show that it is working
    )
    x = torch.randint(0, 20000, (1, 8192)).long()
    torch.onnx.export(model,x,"reformerLM.onnx",verbose=True,opset_version= 14,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)

def reformer():
    model = Reformer(
    dim = 512,
    depth = 12,
    heads = 8,
    lsh_dropout = 0.1,
    causal = True)
    x = torch.randn(1, 8192, 512)
    torch.onnx.export(model,x,"reformer.onnx",verbose=True, opset_version= 14,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)

def lsh():
    model = LSHSelfAttention(
    dim = 128,
    heads = 8,
    bucket_size = 64,
    n_hashes = 8,
    causal = False)
    x = torch.randn(10, 1024, 128)
    torch.onnx.export(model,x,"lsh.onnx",verbose=False)
    
if __name__ == '__main__':
    # Reformer based functions have issues exporting to onnx. 
    # Hence they are listed just for reference purpose.
    lsh()