# Novel Techniques: Composition/Superposition-Informed LM

## Summary

This submission stacks nine novel techniques on the SOTA baseline (thwu1, PR #180), including five inspired by Olah's "Distributed Representations: Composition & Superposition" (2023):

**Architecture techniques:**
1. **TrigramHash Embedding** - 3-token pattern capture via hashed embeddings
2. **Causal Depthwise Convolutions** - Per-block local pattern extraction (~2K params/layer)
3. **Frequency-Tiered Bigram Embeddings** - Dedicated slots for top-256 bigrams (composition), hashing for rare ones (superposition)
4. **relu³ MLP Activation** - Higher sparsity for increased superposition capacity
5. **Per-Head k_gain + v_scale** - Compositional head specialization (120 params total)

**Training techniques:**
6. **Progressive Sequence Length** - Curriculum from 512 -> 1024 -> 2048 tokens
7. **Embedding Orthogonality Regularization** - Encourages tok_emb/bigram/trigram to occupy orthogonal subspaces

**Quantization techniques:**
8. **Mu-Law Companding** - Non-uniform quantization preserving superposition-critical small values
9. **Percentile Clipping** - 99.99th percentile scale for intN quantization

Built on: 10L, 512-dim, 3x MLP, int5/int6 mixed quant, SmearGate, BigramHash(10240), SWA, Muon WD=0.04.

## Composition & Superposition Theory

Inspired by Olah (2023), this submission deliberately applies two orthogonal representation strategies:

- **Composition** (independent features in orthogonal subspaces): Used for high-frequency patterns that benefit from clean separation. The frequency-tiered bigrams give the 256 most common pairs collision-free dedicated embeddings. The orthogonality regularization encourages tok_emb, bigram, and trigram projections to occupy non-interfering subspaces. Per-head scaling enables compositional specialization of attention heads.

- **Superposition** (more features than dimensions via sparse coding): Used for rare/sparse patterns that tolerate interference. Rare bigrams and all trigrams use hash-based superposition. relu³ increases MLP hidden layer sparsity, enabling more features to be packed via superposition. Mu-law companding preserves the small but precise values that superposed features require.

The key insight: **common features need composition (clean separation), sparse features tolerate superposition (dense packing)**. Feature sparsity mediates the optimal trade-off.

## Novel Techniques

### 1. TrigramHash Embedding

Extends BigramHash to capture 3-token patterns. Hashes `(t[i-2], t[i-1], t[i])` into a learned 8192-bucket embedding table (dim=96, projected to 512). Uses different prime multipliers (48271, 31547, 19997) from BigramHash to minimize correlation.

- Parameters: ~835K
- Motivation: BigramHash scaling from 4K to 10K buckets gave -0.002 bpb, suggesting n-gram features are underexploited.

### 2. Causal Depthwise Convolutions

1D causal depthwise convolution (kernel_size=4) before attention in each block. Identity-initialized.

- Parameters: 2048 per layer (20K total)
- Motivation: Extends SmearGate's bigram blending throughout network depth.

### 3. Frequency-Tiered Bigram Embeddings (Composition/Superposition)

Top-256 most frequent byte bigrams ("e ", " t", "th", "he", "in", ...) get dedicated collision-free embedding slots. Remaining bigrams hash into the rest of the table. Common bigrams activate frequently and suffer from hash collision interference (superposition fails for dense features). Dedicated slots give them clean, compositional representations.

- Parameters: 0 additional (same 10240-bucket table, different indexing)
- Expected gain: -0.001 to -0.002 bpb

### 4. relu³ MLP Activation (Superposition)

Replaces `relu(x)²` with `relu(x)³`. Suppresses small activations more aggressively, increasing effective sparsity in the 1536-dim MLP hidden layer. Per the composition/superposition framework, higher sparsity enables more features to be packed via superposition without destructive interference.

- Parameters: 0 additional
- Expected gain: -0.0005 to -0.001 bpb

### 5. Per-Head k_gain + v_scale (Composition)

Adds per-head learnable key scaling (4 params per layer) and value scaling (8 params per layer). Enables attention heads to specialize into different effective temperatures and output magnitudes — increasing their compositional independence.

- Parameters: 120 total (12 per layer × 10 layers)
- Expected gain: -0.0005 to -0.001 bpb

### 6. Progressive Sequence Length Training

Curriculum: seq_len=512 (0-30% time) → 1024 (30-60%) → 2048 (60-100%).

### 7. Embedding Orthogonality Regularization (Composition)

Soft penalty encouraging bigram/trigram projection weights to be orthogonal to tok_emb weights: `λ * ||proj_W @ tok_emb_W^T||²_F`. This encourages the three embedding sources to occupy orthogonal subspaces (composition principle), preventing destructive interference when summed.

- Parameters: 0 additional, λ=0.001
- Expected gain: -0.001 bpb

### 8. Mu-Law Quantization Companding (Superposition)

Applies mu-law companding (μ=15) before int5/int6 quantization. Maps weights through `sign(x) * log(1+μ|x|) / log(1+μ)` before rounding, with inverse at dequantization. Allocates more quantization bins near zero where superposed features encode small but meaningful values, at the expense of large-magnitude resolution.

- Parameters: 0 additional
- Expected gain: -0.001 to -0.002 bpb

### 9. Percentile Clipping for intN Quantization

Uses 99.99th percentile instead of exact max for quantization scale. Clips ~0.01% outlier weights for better dynamic range utilization.

## Architecture

```
Embedding: tok_emb(1024, 512) + BigramHash(10240, 128->512, freq-tiered top-256)
           + TrigramHash(8192, 96->512) + SmearGate + orthogonality reg
Per Block: CausalConv1d(k=4) -> RMSNorm -> GQA Attn(8h/4kv, k_gain, v_scale)
           -> RMSNorm -> MLP(3x, relu³)
Layout: 10 layers, 512 dim, U-Net skips, tied embeddings
Training: Muon WD=0.04, progressive seq_len (512->1024->2048), orth reg λ=0.001
Quant: int5 MLP / int6 attn, mu-law companding (μ=15), percentile clipping, zstd-22
Eval: Sliding window stride=64
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 10 |
| model_dim | 512 |
| mlp_mult | 3.0 |
| train_seq_len | 2048 (final) |
| prog_seq phases | 512/1024/2048 at 0.3/0.6/1.0 time fraction |
| trigram_vocab_size | 8192 |
| trigram_dim | 96 |
| causal_conv_kernel | 4 |
| bigram_vocab_size | 10240 |
| bigram_dedicated_slots | 256 (top-frequency) |
| muon_weight_decay | 0.04 |
| matrix_lr | 0.02 |
| swa_start_frac | 0.4 |
| eval_stride | 64 |
| orth_reg_lambda | 0.001 |
| mulaw_mu | 15.0 |
| activation | relu³ |

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters set as defaults in `train_gpt.py`. No env vars needed.

## Results

*Pending 8xH100 validation*

## Hardware

8x NVIDIA H100 80GB HBM3 SXM (RunPod).
