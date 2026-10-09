# Generates HomeBred-LLM/TestData/test_tiny_moe.gguf (random-weight 4-layer, 8-expert qwen3moe).
# Usage: pip install gguf numpy && python make_tiny_moe.py ../HomeBred-LLM/TestData/test_tiny_moe.gguf
import numpy as np, gguf, sys
out = sys.argv[1]
arch = "qwen3moe"
L, E, K, D, FF, H, HKV, HD = 4, 8, 2, 64, 32, 4, 2, 16
special = ["<unk>", "<s>", "</s>"]
bytes_t = [f"<0x{i:02X}>" for i in range(256)]
pieces = ["▁the", "▁a", "▁hello", "▁world", "▁model", "▁expert", "▁is", "▁to", "▁and", "▁of"]
tokens = special + bytes_t + pieces
V = len(tokens)
w = gguf.GGUFWriter(out, arch)
w.add_name("tiny-moe-test")
w.add_block_count(L); w.add_context_length(2048); w.add_embedding_length(D)
w.add_feed_forward_length(FF); w.add_head_count(H); w.add_head_count_kv(HKV)
w.add_key_length(HD); w.add_value_length(HD)
w.add_rope_freq_base(10000.0); w.add_layer_norm_rms_eps(1e-6)
w.add_expert_count(E); w.add_expert_used_count(K); w.add_expert_feed_forward_length(FF)
w.add_rope_dimension_count(HD)
w.add_tokenizer_model("llama")
w.add_token_list(tokens)
w.add_token_scores([0.0]*3 + [0.0]*256 + [-1.0]*len(pieces))
w.add_token_types([2,3,3] + [6]*256 + [1]*len(pieces))
w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0)
w.add_add_bos_token(True)
rng = np.random.default_rng(0)
def t(name, *shape, scale=0.05):
    w.add_tensor(name, (rng.standard_normal(shape) * scale).astype(np.float32))
t("token_embd.weight", V, D)
for i in range(L):
    p = f"blk.{i}."
    w.add_tensor(p+"attn_norm.weight", np.ones(D, np.float32))
    t(p+"attn_q.weight", H*HD, D); t(p+"attn_k.weight", HKV*HD, D); t(p+"attn_v.weight", HKV*HD, D)
    t(p+"attn_output.weight", D, H*HD)
    w.add_tensor(p+"attn_q_norm.weight", np.ones(HD, np.float32)); w.add_tensor(p+"attn_k_norm.weight", np.ones(HD, np.float32))
    w.add_tensor(p+"ffn_norm.weight", np.ones(D, np.float32))
    # skewed router so routing has structure: experts 0/1 favoured
    gi = (rng.standard_normal((E, D)) * 0.05).astype(np.float32); gi[0] += 0.2; gi[1] += 0.1
    w.add_tensor(p+"ffn_gate_inp.weight", gi)
    t(p+"ffn_gate_exps.weight", E, FF, D); t(p+"ffn_up_exps.weight", E, FF, D); t(p+"ffn_down_exps.weight", E, D, FF)
w.add_tensor("output_norm.weight", np.ones(D, np.float32))
t("output.weight", V, D)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("ok", out)
