# Qwen3.8-27B

`models/qwen3_8_27b/` implements W8A8 Gated DeltaNet and gated-attention
operators, decoder layers, and text prefill/decode entries. Packed prefill
uses separate convolution/recurrent state rows and paged K/V ownership for
each request. Model and serving integration remain under development.

Qwen3.8-27B interleaves three Gated DeltaNet layers with one gated-attention
layer, `LLLA` repeated sixteen times over 64 layers, so 48 of its layers run the
operators in this directory.

## Model configuration

[config.py](../../../models/qwen3_8_27b/config.py) carries the values from the
published `config.json` under their HuggingFace names, so they can be checked
against it without a translation table.

| Property | Value |
| --- | --- |
| Hidden size | 5120 |
| Layers | 64 — 48 linear attention, 16 full attention (`full_attention_interval = 4`) |
| Linear-attention heads | 48 value, 16 QK (`linear_num_value_heads`, `linear_num_key_heads`) |
| Linear-attention head dim | 128 for both key and value |
| Short convolution | depthwise, kernel 4, over all 10240 q/k/v channels |
| Chunk length | 128 — our tiling choice, not a model constant |
| Platform | Ascend A2/A3 (`-p a2a3`); A5 serving is not validated |
| Quantization | W8A8 around the delta rule, which has none itself (no weights): int8 weights per output channel, int8 activations per token. `in_proj_qkv` and `in_proj_z` are int8 both sides; `out_proj` is BF16 both sides, matching the deployed checkpoint, so the block quantizes the hidden states and nothing else. `in_proj_a`, `in_proj_b` and the convolution taps stay bf16 as well, where int8 would move the block output by 3e-5 and save nothing. Measured cost on the real layer at T = 8192: 1.6e-2 against the float64 truth, where quantizing `out_proj` too costs 3.0e-2 |

Grouped-query attention is the shape that matters here: value head `h` reads key
head `h // 3`. q and k carry 16 heads; v, beta, the gate, the state and the
output are per value head.

## Serving interface

As in the DeepSeek model, pypto-lib owns the L3/HOST entries and the kernel
capability manifest. Serving supplies runtime dimensions and buffers; it does
not redefine the kernel signatures. Qwen's factories specialize the fixed
prefill extent, owner count, decode batch and resident cache extents:

| Module | Entry factory | Computation |
| --- | --- | --- |
| [prefill_text_fwd.py](../../../models/qwen3_8_27b/prefill_text_fwd.py) | `build_l3_kernel(t, rows=..., max_pages=..., kv_layer_stride=..., state_count=...)` | Packed text prefill, selected logits and updated caches |
| [decode_fwd.py](../../../models/qwen3_8_27b/decode_fwd.py) | `build_l3_kernel(batch, max_pages=..., kv_rows=..., state_count=..., active_rows=...)` | Decoder layers over stacked weights and layer-specific cache descriptors |
| [embedding.py](../../../models/qwen3_8_27b/embedding.py) | `build_l3_kernel(batch)` | Token embedding |
| [rms_lm_head.py](../../../models/qwen3_8_27b/rms_lm_head.py) | `build_l3_kernel(batch)` | Final normalization and vocabulary logits |

Each factory returns a `pl.jit.host` entry calling the existing L2 compute
implementation. The forward entries declare convolution, recurrent and K/V
cache tensors as `pl.InOut`; serving keeps these buffers device-resident.
Weights stack on the row axis in global-layer order for common tensors,
GDN-layer order for linear-attention tensors, and attention-layer order for
full-attention tensors. The caller owns request metadata and cache allocation.

[serving_contract.py](../../../models/qwen3_8_27b/serving_contract.py) is
importable without PyPTO and describes scheduling and shape limits. It requires
exact fused prefill extents: padding must not update a live recurrent state.
Fused extents are multiples of 2048 tokens, up to 32768, and each packed owner
contributes spans aligned to 256 tokens. Decode batches are padded to a multiple
of 16, at most 128 rows. These are implementation limits; the complete serving
path has been validated at a 2048-token extent with two concurrent requests.

## The operators

The layer's core is the chunked delta rule, split the way the reference
PTO-ISA implementation splits it. Each file is a standalone entry point. The
first six are the delta rule; the rest are the block around it, in the order
the block runs them.

| Operator | What it computes |
| --- | --- |
| [chunk_cumsum.py](../../../models/qwen3_8_27b/chunk_cumsum.py) | chunk-local prefix sum of the gate logits |
| [scaled_dot_kkt.py](../../../models/qwen3_8_27b/scaled_dot_kkt.py) | the gated key-key matrix `A`, strictly lower triangular |
| [solve_tril.py](../../../models/qwen3_8_27b/solve_tril.py) | `(I + A)^-1` per chunk, with FP32 diagonal inversion and block substitution |
| [wy_fast.py](../../../models/qwen3_8_27b/wy_fast.py) | the WY representation, `W` and `U` |
| [chunk_h.py](../../../models/qwen3_8_27b/chunk_h.py) | the inter-chunk state recurrence and `V_new` |
| [chunk_o_pipe.py](../../../models/qwen3_8_27b/chunk_o_pipe.py) | the chunk output, inter-chunk plus intra-chunk, as an explicit cube/vector pipeline |
| [quant_x.py](../../../models/qwen3_8_27b/quant_x.py) | per-token int8 quantisation of the hidden states, shared by the projections that read them |
| [in_proj_qkv.py](../../../models/qwen3_8_27b/in_proj_qkv.py) | hidden states to q, k and v, A8W8 |
| [in_proj_z.py](../../../models/qwen3_8_27b/in_proj_z.py) | hidden states to the gated norm's gate, A8W8 |
| [in_proj_ab.py](../../../models/qwen3_8_27b/in_proj_ab.py) | hidden states to `a` and `b`, the two fused into one bf16 matmul |
| [short_conv.py](../../../models/qwen3_8_27b/short_conv.py) | the depthwise causal convolution over the q/k/v channels, with silu |
| [qk_norm_gate.py](../../../models/qwen3_8_27b/qk_norm_gate.py) | the q/k L2-norm and scaling, and `beta` and the decay gate from the a/b projections |
| [gated_rmsnorm.py](../../../models/qwen3_8_27b/gated_rmsnorm.py) | the gated RMSNorm after the delta rule, in bf16 for the block and with a per-token INT8 epilogue for an A8W8 reader |
| [bf16_linear.py](../../../models/qwen3_8_27b/bf16_linear.py) | the normed output back to the residual stream, BF16 both sides |
| [gdn_layer.py](../../../models/qwen3_8_27b/gdn_layer.py) | the delta rule's six stages as one program |
| [gdn_block.py](../../../models/qwen3_8_27b/gdn_block.py) | all fourteen as one program, hidden states in and out |
| [prefill_layer.py](../../../models/qwen3_8_27b/prefill_layer.py) | a whole decoder layer over a prefill chunk, either mixer, with both residual adds and the MLP |

The two A8W8 projections are the same arithmetic at two shapes and share
one implementation, [a8w8_linear.py](../../../models/qwen3_8_27b/a8w8_linear.py);
each instance file supplies its shape, its tiles and where its weight and
references come from.

`solve_tril` is the delta-rule triangular inversion of
[arXiv:2605.21325](https://arxiv.org/abs/2605.21325).

The inverse uses 16-token diagonal blocks and keeps the inversion and block
substitution in FP32, narrowing only the published inverse to FP16. This
preserves precision for correlated keys, including repeated-token prompts.
The default `solve_tril.py` validation includes a correlated-key case alongside
its random cases; select it with `--input-case correlated --seq-len 256`.
The diagonal and lower-substitution tasks use 32 and 4 cross-core transfer
slots, respectively, with plain sequential loops and unchanged arithmetic.
These depths pass random, correlated and checkpoint solve gates, composed
random/checkpoint block gates with fresh and resumed state, and full-stack
logit/cache equivalence on the two-request serving configuration.

The timing figures below predate this FP32 solve update and require
remeasurement before they describe the current implementation.

## The layer

[gdn_layer.py](../../../models/qwen3_8_27b/gdn_layer.py) composes all six as one
forward. `A`, `A_inv`, `W`, `U`, the per-chunk state snapshots and `V_new` are
allocated inside it and never leave; only q, k, v, the gate and beta come in, and
only the output goes out. It is scored end to end against the float64 reference
rather than stage by stage.

The standalone and composed paths use the same operators. Both publish the
triangular inverse as FP16, which is the dtype consumed by `wy_fast`;
intermediate inversion arithmetic remains FP32.

`gdn_layer.py` is the delta rule alone. The surrounding block is
[gdn_block.py](../../../models/qwen3_8_27b/gdn_block.py), below.

## The block

[gdn_block.py](../../../models/qwen3_8_27b/gdn_block.py) runs all fourteen
operators as one program: bf16 hidden states in, bf16 hidden states out, with
every intermediate allocated inside and never reaching the caller. The delta
rule inlines into it as one callee in turn, so the six stages inline into
`gdn_layer` and `gdn_layer` inlines here.

Three interfaces are shaped by the composition rather than by any one operator,
and each is answered where it costs nothing:

- The convolution reads the K-1 tokens before each of its windows, so
  `in_proj_qkv` takes a `row_off` and writes straight into the padded buffer
  the convolution wants; the rows above it are zeroed by a ten-task fill.
  Nothing copies `[T, C]` into `[T + K - 1, C]`.
- The convolution writes q, k and v as three tensors rather than one, so each
  consumer reads the rows of a contiguous buffer -- and v, which goes straight
  to the delta rule, is written in fp16 at the store instead of by a later
  cast pass.
- The gated norm declares `z` in the `[T, H*D]` layout `in_proj_z` writes.

No slicing is needed at any call site: every delta-rule kernel already
reshapes its rank-3 inputs to `[T, heads*D]` as its first statement, so what
the consumers need is each tensor contiguous in its own buffer, not a
particular rank.

**On a2a3 at T = 8192: 9353 us**, against 9559 for the same fourteen kernels
timed separately in the same grant -- so composing is worth about 205 us, and
that difference is the whole of the inter-kernel overlap the runtime performs.
Both figures come from one sitting, which is what makes the difference
meaningful: the block's own between-grant spread is about 150 us, and it
re-measured 9378 us mean over 50 rounds on a later one.
Most of it is at the head, where `in_proj_ab`, the halo fill, the gate half of
`qk_norm_gate` and `chunk_cumsum` all run underneath `quant_x` for free.

**Accuracy is scored as ratios against the quantization floor, not as an
element-wise tolerance**, because the block quantizes the hidden states per
token: a difference far below the int8 step flips values that sit near a
rounding boundary, and every projection downstream then reads a different
integer -- so an element-wise comparison taken after a quantiser measures
which side of a boundary a value fell on, not whether the kernel is right.
The gate is instead total relative error within 1.05x of what W8A8 alone
costs and worst token row within 1.25x, plus the same two distances to the
fake-quant chain itself. Measured at T = 2048 on random weights: **1.032x and
1.094x**, and 0.272 of the floor away from the chain the kernels implement.
The worst row is the sharper of the two -- one dead row in 2048 moves the
whole-tensor norm by 1.12x and the worst row by 10.6x.

Those ratios have a smaller denominator than they used to. The same
measurement on the A8W8 `out_proj` reads 1.006x and 0.987x at a floor of
4.50e-02, where the BF16 one sits at 1.99e-02: the kernels' own fp16 and bf16
rounding is unchanged in absolute terms, so it is a larger fraction of a
floor half the size. In absolute terms the block moved **closer** to the
chain it implements, 1.88e-02 to 5.43e-03.

Still out of scope here: variable-length batches (one sequence, `B = 1`).

## The decoder layer

[prefill_layer.py](../../../models/qwen3_8_27b/prefill_layer.py) wraps either
mixer in the rest of a decoder layer: the input norm, the mixer, the
post-attention norm and the MLP. Each norm is `rmsnorm`'s fused add-and-norm,
which reads the running residual and the previous stage's delta and writes both
the new residual and the normed stream, so **the layer boundary sits one
addition behind the module boundary**: `(residual, delta)` in,
`(residual_out, mlp_out)` out. The residual runs in FP32 and everything else in
BF16 -- bf16 has eight mantissa bits, and a 64-layer accumulation would spend
half of them. `--kind gdn` composes `gdn_block`, `--kind attn` composes
`gated_attention`, and both are gated against `reference.residual_layer`, which
takes the mixer as a callable and serves the decode path too.

**The chunk a layer accepts, and what it costs** (a2a3, one card, one grant, a
real layer's weights resident, five rounds after two warmup, median):

| tokens | ring 1 | GDN layer | per token | attention layer | per token |
|---|---|---|---|---|---|
| 2048 | 4 GiB | | | 4.83 ms | 2.358 us |
| 8192 | 4 GiB | 19.56 ms | 2.388 us | 19.73 ms | 2.409 us |
| 16384 | 8 GiB | 40.41 ms | 2.467 us | 49.84 ms | 3.042 us |
| 32768 | 16 GiB | 82.27 ms | 2.511 us | 139.27 ms | 4.250 us |
| 65536 | 32 GiB | 171.03 ms | 2.610 us | 487.82 ms | 7.443 us |
| 98304 | 32 GiB | 258.08 ms | 2.625 us | | |

The attention layer needs less: 4 GiB carries it to 16384 and 8 GiB to 32768.
Each heap above 4 GiB in that table was taken because the half-size heap
**failed mid-run** -- as a generic AICPU exception, not as an allocation error,
so a short ring heap looks like a device fault. `ring_heap` is per ring and must
be a power of two, so the four-entry list raises ring 1 alone and leaves the
other three at their defaults; 98304 was not probed below 32 GiB, and 4 GiB is
the entry's own default rather than a measured floor.

The GDN layer is **linear** in the token count, 2.39 to 2.63 us per token over a
twelvefold range, and the attention layer is **quadratic**, 2.36 to 7.44. The
chunk is therefore chosen by the attention path and by what the ring heap leaves
for the resident weights, not by the GDN path.

## The short convolution

[short_conv.py](../../../models/qwen3_8_27b/short_conv.py) is the depthwise
causal convolution that follows the q/k/v projection: four taps over the token
axis per channel, fp32 accumulate, silu fused in the epilogue, no bias.

Its input is `[T + K - 1, C]`, carrying the K-1 tokens before the sequence in
its first rows. That is what keeps every tap offset non-negative, so no block
needs a head-padding special case; for prefill those rows are zero, and they
are where a decode cache would sit.

On a2a3 at T = 8192 over all 10240 channels, 50 rounds: **648 us**, 517 GB/s of
GM traffic against a measured copy roof of 1236 GB/s. CANN's own depthwise
`conv1d` plus silu takes 1532 us channels-first and 2597 us once the transpose
our token-major layout would need is charged, so 2.4x and 4.0x. Accuracy
against the float64 reference is max abs 2.6e-02 on the real layer, inside the
6e-02 the hand-written PTO-ISA kernel's own tests use for bf16.

## The qk-norm and gate

[qk_norm_gate.py](../../../models/qwen3_8_27b/qk_norm_gate.py) produces the
four things the delta rule reads: q and k L2-normalised over the head dim with
the epsilon inside the square root and q scaled by `D^-0.5`, `beta` as a
sigmoid of one small projection, and the decay `g` from the other.

`beta` comes out head-major, `[H, T]`, because that is what `wy_fast` and
`scaled_dot_kkt` declare; the kernel transposes on chip rather than leaving it
to the host. `softplus` is written as `log(1 + exp(x))` with the exponent
clamped, since the naive form overflows fp32 and this model's `dt_bias`
already reaches 19.25.

On a2a3 at T = 8192, 50 rounds: **122 us**, 1135 GB/s against a measured copy
roof of 1236 GB/s, so this kernel is bandwidth-bound. Accuracy against the
float64 reference is 1.6e-03 relative Frobenius on q and k, and better on
`beta` and `g`.

## The gated RMSNorm

[gated_rmsnorm.py](../../../models/qwen3_8_27b/gated_rmsnorm.py) is the norm
that follows the delta rule, `y = o * rsqrt(mean(o^2) + eps) * norm_w * silu(z)`
per (token, head) over the head dim, with the cast sequence the model's own
`Qwen3_5RMSNormGated` uses: fp32 variance and rsqrt, narrow to the model dtype,
apply the weight there, apply the gate in fp32.

The block takes the form that ends in the module's own cast, `out_proj` being
bf16 both sides. The file also carries an epilogue that quantises the result
per token over the whole 6144-wide row -- what an A8W8 reader of this output
needs, which is how the block read it before the BF16 `out_proj` -- and
`--bf16-out` selects between the two.

On a2a3 at T = 8192, 50 rounds: **398 us** for the bf16 form the block uses
(759 GB/s of GM traffic) and **643 us** with the int8 epilogue (391 GB/s). The comparison is a `torch_npu`
composition -- `npu_rms_norm`, eager silu and multiply, `npu_dynamic_quant`,
there being no gated-norm op -- at 2584 and 2431 us, so 4.0x and 6.1x. Accuracy against the
float64 reference is 3.3e-2 with the epilogue and 3.1e-3 without, so the
quantisation is the entire gap and the kernel adds nothing measurable to it.

## The projections

Two of the four projections around the delta rule are W8A8: `in_proj_qkv` and
`in_proj_z`, with int8 weights at one scale per output channel, int8
activations at one scale per token, INT32 accumulation over a pipelined K loop,
dequantised by the two scales and written as bf16. The other two are bf16 --
`in_proj_ab` because int8 there buys nothing (below), and `out_proj` because
the deployed checkpoint publishes that weight unquantized.

[quant_x.py](../../../models/qwen3_8_27b/quant_x.py) does the activation side
once for everything that reads the hidden states, and it is the block's only
activation quantisation. The gated norm has the same arithmetic available in
its own epilogue, for an A8W8 reader of its output; the block does not use it.

[a8w8_linear.py](../../../models/qwen3_8_27b/a8w8_linear.py) is the matmul.
Its one structural choice is that a block walks an `m_group` by `n_group`
**patch** of output tiles rather than a single tile. One output tile is capped
near 128x128 by the cube-to-vector crossing -- the cross-core ring is sized by
the INT32 accumulator and charged to the same vector buffer that holds the
working copy -- and at one tile per block `in_proj_qkv` is 5120 tasks, more
than the runtime's ring heap carries. A patch fixes both: it cuts the task
count and it is what a bigger tile would have bought, since a block reads
`(m_group * m_tile + n_group * n_tile) * K` bytes for
`m_group * n_group` tiles of output. Square patches win; the shape alone is
worth 1.4x at identical tiles.

The cap is on the tile that *crosses*, though, not on what the cube may work
on, so the inner loop takes two n-tiles against one activation tile and drains
them separately -- the cube sees a 128x256 region from a single L1 operand
while each crossing stays 128x128. That is worth another 6 to 8%.

[in_proj_ab.py](../../../models/qwen3_8_27b/in_proj_ab.py) stays bf16 and fuses
`in_proj_a` and `in_proj_b` into one `[96, 5120]` weight: N = 48 alone is legal
under the multiple-of-16 tile rule but a poor cube tile, and two passes over
the hidden states would cost twice what one does.

On a2a3 at T = 8192, 50 rounds, against `torch_npu` 2.10.0 on the same card:

| kernel | PyPTO us | CANN operator | CANN us |
| --- | ---: | --- | ---: |
| `quant_x` | 191 | `npu_dynamic_quant` | 105 |
| `in_proj_qkv` | 1813 | `npu_quant_matmul` | 1660 |
| `in_proj_z` | 1014 | `npu_quant_matmul` | 993 |
| `in_proj_ab` | 115 | `F.linear` bf16 | 47 |

The two GEMMs run at 474 and 508 INT8 TOPS, about 1.7x their own bf16
`F.linear` at the same shapes and 0.92 to 0.98x of CANN's dedicated W8A8
operator. A
control that drops the dequantisation entirely measures within the noise of the
full kernel, so that gap is the cube schedule under a 128x128 output tile, not
the epilogue. Accuracy: each GEMM sits 1.7e-03 from the float64 W8A8 chain it
implements -- bf16 output rounding and nothing else -- and the quantisation is
the whole of the rest.

The block's `out_proj` left that table when it became BF16. One card, one
grant, T = 8192, 50 rounds: **939 us** as A8W8 against **1858 us** through
`bf16_linear`, 1.98x. The remaining gap is the dtype and not the schedule: the
patch shape and the K tile were swept at this shape, ten points, and nothing
beat the current 8 x 2 -- the shapes that cut requested GM traffic 2.5 to 3.3x
measured 11 to 41% **slower**, so the extra bytes are not what costs it. What
did help was latency, the two knobs its INT8 sibling already had: a GM
transport ring two slots deep with the on-chip reserve at one, and the K
stagger. Together they take the projection from 1995.6 to **1857.9 us**
(-6.9%; depth 2 alone -3.4%, the stagger alone -2.1%, depth 4 no further) and
the composed block from 9720.9 to **9637.0**. A smaller crossing at `n_tile`
64 with two slots is 34% worse, so the depth is worth more than the tile it
would cost.

On the composed block, A8W8 against the tuned BF16 in one grant: **9094.7
against 9637.0 us at T = 8192**, so the projection costs **6.0% of the block**
and buys the 1.9x smaller quantisation error above.

## Packed prefill

[prefill_fwd.py](../../../models/qwen3_8_27b/prefill_fwd.py) composes three GDN
layers and one full-attention layer per group. Its packed requests share
weights and token-local projections, with separate sequence boundaries and
cache owners. [prefill_text_fwd.py](../../../models/qwen3_8_27b/prefill_text_fwd.py)
adds embedding lookup and scores the final token of each packed request.

The stack takes these INT32 descriptors, where `R` is the number of requests:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `row_base` | `[T]` | First packed token index of the token's request |
| `row_owner` | `[T]` | Request index for each token |
| `row_ends` | `[R]` | Exclusive packed end of each request |
| `state_rows` | `[gdn_layers * R]` | Pool row at `layer * R + owner`, shared by convolution and recurrent state |
| `page_bases` | `[R]` | Number of cached 128-token pages before each current request chunk |
| `visible_pages` | `[R]` | Cached plus current pages for each request |
| `block_table` | `[R, max_pages]` | Logical-to-physical KV page map for each request |

`kv_layer_stride` gives the physical pool rows between attention layers. State
rows must be distinct across active layer/request pairs; current KV writes
must not overlap another request's pages. Fresh requests need zero incoming
GDN state. Resumed requests supply their own convolution halo, recurrent state,
KV prefix, and absolute-position RoPE tables.

Current request lengths and cached prefixes must align to 256 tokens. The
total packed extent must be a multiple of 2048 tokens for the projection
schedule. Padding tokens through GDN updates state, so callers must handle
remainders through decode until valid-token masking is supported.

These cases use 256 and 1792-token requests at T = 2048. A page base of 4
resumes a request after 512 cached tokens; `--resume` initializes that
request's GDN state as well. Run them on a device assigned by the site's
allocator:

```bash
python models/qwen3_8_27b/prefill_fwd.py -p a2a3 -d "$TASK_DEVICE" --packed --groups 1
python models/qwen3_8_27b/prefill_fwd.py -p a2a3 -d "$TASK_DEVICE" --packed --groups 1 --page-bases 0,4 --resume
python models/qwen3_8_27b/prefill_text_fwd.py -p a2a3 -d "$TASK_DEVICE" --packed --groups 1 --page-bases 0,4 --resume
python models/qwen3_8_27b/prefill_fwd.py -p a2a3 -d "$TASK_DEVICE" --packed --groups 2 --page-bases 2,4 --resume
python models/qwen3_8_27b/prefill_text_fwd.py -p a2a3 -d "$TASK_DEVICE" --packed --groups 2 --page-bases 0,4 --resume
python models/qwen3_8_27b/prefill_text_fwd.py -p a2a3 -d "$TASK_DEVICE" --packed --groups 16 --page-bases 0,4 --resume
```

The oracle runs each request independently through the float64 and fake-quant
stack chains. Gates check each request's output streams and each layer/request
cache region against its own quantization floor, preserve untouched pool rows
exactly, and exercise the following decode consumer. Projected K/V values use
numerical tolerances. The text gate scores only the first `valid_rows` logit
rows; the kernel leaves the padded tail undefined.

The four-layer cases, the eight-layer resumed stack and mixed text cases,
and the full 64-layer mixed text case pass these gates on a2a3 with synthetic
weights. The full-depth gate covers both requests' final-token logits, both
streams, every cache layer/owner, and subsequent decode consumers. This does
not establish real-checkpoint serving correctness or served performance.

The text entry uses `text_ring_heap(T)` for its runtime buffers. At multiple
groups, the retained loop places whole-layer buffers on ring 3; the helper
sizes it with the same layer budget as rings 1 and 2. Larger text chunks need
their own capacity and correctness checks.

The pypto-serving Qwen3.8 runner packs complete 256-token request spans into
these launches, maps owner-major physical state slots to the kernel's
layer-major descriptors, and restores logits to batch order. Unaligned
positions and tails use decode without synthetic padding. Its full-depth
two-request fused path passes two a2a3 NPU runs against both saved oracle
sets, including output streams, caches and host decode consumers. That gate
uses synthetic weights and resets cache pools between runs. A separate
real-checkpoint serving suite matches vLLM-ascend v0.23.0 on four raw prompts
of 5, 5, 2048 and 2050 tokens, with eight greedy outputs each over two repeats:
all 64 generated token IDs match. It uses two sequences, context 4096,
2048-token scheduling, 128-token pages and disabled prefix caching. The
FP32 solve passes its unchanged pointwise and Frobenius gates, including a
correlated-key regression. Performance against the default vendor configuration
must be remeasured after kernel changes.

## Validating and benchmarking

Each operator file is its own test. It owns the float64 torch reference for its own
stage, builds its inputs by calling the references of the stages before it, computes a
golden, and scores the device result with the harness's `ratio_reldiff` per point plus a
relative Frobenius norm over the whole output.
[reference.py](../../../models/qwen3_8_27b/reference.py) holds what they share: the
drawn inputs, the two layout conversions and the acceptance criterion.
A default invocation runs each entry's `DEFAULT_CASES`: a single chunk where the entry
supports one, the cross-chunk recurrence, and the minimum dispatch. `--seq-len` runs one
shape instead, and `BENCH_SEQ = 8192` is the prefill shape a performance run names.
`PYPTO_BENCH=1` turns any of them into a benchmark, and the harness prints the median
`effective_us`.

Around those six, `reference.py` carries the whole block: the four projections, the
depthwise convolution, the qk-norm and gate, the gated RMSNorm and `out_proj`, read
line by line from `Qwen3_5GatedDeltaNet` in `transformers`. That block chain is what
every operator and the composed block are scored against, so it
is itself checked against the model's own module:
[test_block_reference.py](../../../models/qwen3_8_27b/test_block_reference.py)
runs `Qwen3_5GatedDeltaNet` on the same hidden states and weights and compares
the two at every projection, after the delta rule, after the norm and at the
output. With the module in float64 the projections agree exactly and everything
after the gate to about 3e-7 -- the module computes its gate, delta rule and norm
statistics in fp32 by its own code, which is the floor. In the model's own bf16
the block output sits 3e-3 to 7e-3 from the float64 chain, which is the
yardstick for what a device-side block can be held to.

The reference is two references in one. On bf16 weights it is the float64
truth above. On int8 weights (`reference.quantize_weights`, the same
per-output-channel chain `models/qwen3_14b` and `models/deepseek_v4_pro` use)
with per-token quantisation of the hidden states and of the normed output, it
is the W8A8 chain the projection kernels will implement, with their exact
arithmetic. `test_block_reference.py --quant` reports what each part of that
scheme costs against the truth: on the real layer at T = 8192 the whole scheme
sits 3.0e-2 away, of which int8 weights alone are 1.1e-2, the per-token hidden
states 1.3e-2 and the per-token normed output 2.4e-2 — ten times the model's own
bf16 rounding. That is why the projection kernels are gated against this chain
and only reported against the truth: quantisation is the whole gap by
construction, and a kernel gate has to sit below it.

Two weight sets serve that check and every later one: random weights drawn the
way the module's own `__init__` draws them, and one real layer.
[weights.py](../../../models/qwen3_8_27b/weights.py) fetches a layer's nine
`linear_attn` tensors from the published checkpoint with a single 221 MiB range
request -- they are contiguous in their shard -- so no checkpoint download is
needed. A trained layer's `A_log` and `dt_bias` reach a decay range the random
init does not, and that is the range where a wrong scale, a wrong eps or a
saturating gate would show.

```bash
python models/qwen3_8_27b/gdn_block.py -p a2a3 -d 0
python models/qwen3_8_27b/gdn_block.py -p a2a3sim --seq-len 2048
python models/qwen3_8_27b/gdn_layer.py -p a2a3 -d 0
python models/qwen3_8_27b/chunk_h.py -p a2a3 --sim
python models/qwen3_8_27b/solve_tril.py -p a2a3 --sim --seq-len 1024
PYPTO_BENCH=1 python models/qwen3_8_27b/gdn_layer.py -p a2a3 -d 0 --seq-len 8192
```

The block is marked device-only for CI: on a simulator it takes about eight
minutes at T = 8192, more than the shared simulator job can spend on one file,
so CI exercises it on hardware and the operators individually on the simulator.
By hand it runs on either, and `--seq-len 2048` is the short shape used for
simulator checks.

The block reference check needs no NPU, only `transformers` (5.17.0), and skips with a message if it is absent. Its default sequence length is 1024, which checks the same thing element for element; the numbers quoted above are at 8192. `weights.py` needs `--fetch` the first time, since it pulls from the checkpoint:

```bash
python models/qwen3_8_27b/weights.py --layer 0 --fetch
python models/qwen3_8_27b/weights.py --quantize
python models/qwen3_8_27b/test_block_reference.py --seq-len 8192
python models/qwen3_8_27b/test_block_reference.py --seq-len 8192 --weights build_output/qwen3_8_27b/linear_attn_layer0.pt
python models/qwen3_8_27b/test_block_reference.py --seq-len 8192 --weights build_output/qwen3_8_27b/linear_attn_layer0.pt --quant
python models/qwen3_8_27b/test_block_reference.py --module-dtype float64,bfloat16
```

These are entry points, not pytest cases: the device ones need an NPU and a
compile, the block check a `transformers` install and a weight download, so they
carry no `test_` functions and CI never collects them.

## Composed decode serving entry

`decode_fwd.build_kernel` composes consecutive groups of three GDN layers and one
full-attention layer. The serving adapter uses all sixteen groups in one NPU
submission, between the existing embedding and final-head programs. Inputs use
stacked weights, layer-major state-row mappings and per-layer attention page
tables; cache pools remain resident and inactive rows use private scratch slots.
The caller must keep `group_start` and the selected group count within the stacked
weight extent.

The optional `active_rows` build argument restricts the existing GDN recurrence
to a live prefix, while matrix and convolution stages keep their padded extent.
Inactive recurrence outputs are zeroed and their state rows remain untouched.
The default, zero, processes all padded rows. Serving selects one- and two-row
variants with a two-head recurrence tile, and retains the padded decoder for
larger active batches. Runtime argument shapes and cache ownership are unchanged.
The standalone GDN block exposes `--active-rows` for reference validation.

The composition reuses the decode layer operators without changing arithmetic or
precision. Validation against the per-layer NPU path covers random and real
weights, nonzero group offsets, the full 64-layer stack, all cache contents and
untouched rows. Full-logit and serving checks cover reordered owners, slot reuse
and exact-token prefill tails. The device validation configuration uses a padded
batch of sixteen with two active requests on Ascend 910B2. Wider serving
configurations need their own end-to-end validation.

Dynamic-offset weight and descriptor views use explicit constant-size `pl.slice`
calls. This preserves the child JIT argument shape on toolchains whose subscript
metadata inference falls back to the full parent extent.
