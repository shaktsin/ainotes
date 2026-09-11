# Day 3 — Building a Decoder Transformer Block from Scratch

**Status:** Ready

**Estimated time:** 6–8 focused hours

Today you will construct the central reusable unit of a decoder-only language model.

The final block will have this structure:

```text
input x [B,T,C]
  │
  ├─ RMSNorm
  ├─ causal multi-head self-attention
  └─ residual addition: x = x + attention(...)
  │
  ├─ RMSNorm
  ├─ SwiGLU feed-forward network
  └─ residual addition: x = x + feed_forward(...)
  │
output [B,T,C]
```

By the end, you should be able to:

- Explain what `nn.Module` and `nn.Parameter` do.
- Read the weight shapes and parameter count of a linear layer.
- Implement RMSNorm.
- Produce queries, keys, and values from token representations.
- Implement scaled dot-product causal attention manually.
- Split attention into multiple heads and recombine them.
- Implement rotary position embeddings (RoPE).
- Implement a SwiGLU feed-forward network.
- Explain pre-normalization and residual connections.
- Assemble and test one complete transformer block.
- Detect shape errors, future-token leakage, and broken gradients.

The goal is not to memorize code. The goal is to understand why each line exists and which tensor shape it produces.

## 1. Day 2 bridge: follow one batch

Keep this shape pipeline visible while working:

```text
raw text
  ↓ tokenizer
token IDs [B,T]
  ↓ token embedding
token vectors [B,T,C]
  ↓ transformer blocks
contextual vectors [B,T,C]
  ↓ output projection
logits [B,T,V]
  ↓ cross-entropy with shifted targets [B,T]
scalar loss []
```

The transformer block receives and returns `[B,T,C]`.

- `B`: batch size—the number of sequences processed together.
- `T`: sequence length—the number of token positions per sequence.
- `C`: model dimension—the number of floating-point features representing each token.
- `V`: vocabulary size—the number of possible token IDs.

The block does not receive token IDs directly. The embedding layer has already converted IDs into floating-point vectors.

```python
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

B, T, C, V = 4, 16, 64, 256

token_ids = torch.randint(0, V, (B, T), device=device)
embedding = nn.Embedding(V, C).to(device)
x = embedding(token_ids)

assert token_ids.shape == (B, T)
assert x.shape == (B, T, C)
assert token_ids.dtype == torch.long
assert x.is_floating_point()
```

### Checkpoint 1 — Shape bridge

Answer before expanding the solution:

1. What does `x[2, 5]` represent?
2. What is its shape?
3. What does `x[:, 5]` represent?
4. What is its shape?

<details>
<summary>Solution</summary>

- `x[2,5]` is the model vector for token position 5 in batch sequence 2. Its shape is `[C]`.
- `x[:,5]` contains position 5 from every batch sequence. Its shape is `[B,C]`.

```python
assert x[2, 5].shape == (C,)
assert x[:, 5].shape == (B, C)
```

</details>

## 2. What `nn.Module` provides

PyTorch models subclass `nn.Module`:

```python
class DoubleAndShift(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * x + self.shift
```

Important pieces:

- `__init__` creates persistent layers and trainable state.
- `forward` defines the calculation.
- Calling `module(x)` invokes the module machinery and then `forward`.
- `nn.Parameter` is a tensor automatically registered as trainable state.
- Child modules assigned as attributes are registered recursively.
- `.to(device)` moves registered parameters and buffers.
- `.train()` and `.eval()` set training behavior for layers such as dropout.
- `.state_dict()` provides values to save in a checkpoint.

```python
module = DoubleAndShift(4).to(device)
sample = torch.ones(2, 4, device=device)
result = module(sample)

print(result)
print(list(module.named_parameters()))
print(module.state_dict().keys())
```

### Ordinary tensor versus parameter versus buffer

```python
class StateExample(nn.Module):
    def __init__(self):
        super().__init__()

        self.ordinary_tensor = torch.ones(3)
        self.trainable = nn.Parameter(torch.ones(3))
        self.register_buffer("persistent_constant", torch.ones(3))
```

| Kind | Trained? | In `state_dict`? | Moved by `.to()`? |
|---|---:|---:|---:|
| Ordinary tensor attribute | No | No | No |
| `nn.Parameter` | Yes | Yes | Yes |
| Registered buffer | No | Yes | Yes |

RoPE frequencies or a fixed causal mask can be stored as buffers, although today we will generate them during the forward pass for clarity.

### Why not create layers inside `forward`?

This is wrong:

```python
class BrokenLayer(nn.Module):
    def forward(self, x):
        projection = nn.Linear(x.size(-1), x.size(-1))
        return projection(x)
```

Every call creates new random weights. They are not stable registered parameters and the optimizer cannot reliably train them.

Create layers in `__init__`:

```python
class StableLayer(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, x):
        return self.projection(x)
```

## 3. Linear layers are learned projections

A linear layer calculates:

$$
y = xW^T + b
$$

Create one:

```python
layer = nn.Linear(
    in_features=64,
    out_features=192,
    bias=False,
)

print(layer.weight.shape)  # [192,64]
```

PyTorch stores the weight as `[out_features,in_features]`. For input `[B,T,64]`, it applies the same projection independently to every token:

```python
x = torch.randn(4, 16, 64)
y = layer(x)

assert y.shape == (4, 16, 192)
```

Parameter count without bias:

```text
64 × 192 = 12,288 parameters
```

With bias, add 192 parameters.

Verify `nn.Linear` manually:

```python
layer = nn.Linear(4, 6, bias=True)
x = torch.randn(2, 3, 4)

automatic = layer(x)
manual = x @ layer.weight.T + layer.bias

torch.testing.assert_close(automatic, manual)
```

The layer changes only the final dimension:

```text
[B,T,C_in] → [B,T,C_out]
```

### Checkpoint 2 — Linear-layer accounting

A projection maps `[B,T,512]` to `[B,T,1536]` with bias disabled.

1. What is the weight shape?
2. What is the output shape when `B=8` and `T=128`?
3. How many parameters does it contain?

<details>
<summary>Solution</summary>

```text
weight shape: [1536,512]
output shape: [8,128,1536]
parameters:   1536 × 512 = 786,432
```

</details>

## 4. Why normalization is needed

Deep networks repeatedly transform and add tensors. Their magnitude can grow, shrink, or become unstable across layers.

Normalization controls the scale of each token vector. Transformer normalization normally operates independently on the final `C` dimension:

```text
input:  [B,T,C]
reduce over: C
output: [B,T,C]
```

RMSNorm uses the root mean square:

$$
\operatorname{RMS}(x)
=
\sqrt{\frac{1}{C}\sum_{i=1}^{C}x_i^2 + \epsilon}
$$

and:

$$
\operatorname{RMSNorm}(x)
=
\frac{x}{\operatorname{RMS}(x)} \odot w
$$

`w` is one learned scale value per channel. RMSNorm does not subtract the mean, unlike LayerNorm.

## 5. Implement RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform the reduction in float32 for numerical stability when x is
        # float16 or bfloat16.
        mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(mean_square + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight
```

Follow the shapes:

```python
B, T, C = 2, 3, 4
x = torch.randn(B, T, C)

mean_square = x.pow(2).mean(dim=-1, keepdim=True)
print(mean_square.shape)  # [B,T,1]

norm = RMSNorm(C)
y = norm(x)
print(y.shape)            # [B,T,C]
```

Broadcasting expands:

```text
x:           [B,T,C]
mean_square: [B,T,1]
weight:          [C]
output:      [B,T,C]
```

### RMSNorm properties

With the learned weight initialized to one, each token’s mean squared output should be approximately one:

```python
torch.manual_seed(42)
x = torch.randn(2, 3, 8)
norm = RMSNorm(8)
y = norm(x)

output_mean_square = y.float().pow(2).mean(dim=-1)
torch.testing.assert_close(
    output_mean_square,
    torch.ones_like(output_mean_square),
    atol=1e-4,
    rtol=1e-4,
)
```

Scaling the input by a positive constant should barely change the normalized result:

```python
torch.testing.assert_close(
    norm(x),
    norm(10 * x),
    atol=1e-4,
    rtol=1e-4,
)
```

Compare with PyTorch’s implementation when available:

```python
reference = nn.RMSNorm(8, eps=1e-6)
reference.weight.data.copy_(norm.weight.data)

torch.testing.assert_close(
    norm(x),
    reference(x),
    atol=1e-5,
    rtol=1e-5,
)
```

### Checkpoint 3 — Break RMSNorm intentionally

Change:

```python
mean(dim=-1, keepdim=True)
```

to:

```python
mean(dim=1, keepdim=True)
```

The code may still run. Explain why its meaning is wrong.

<details>
<summary>Solution</summary>

`dim=1` normalizes across token positions, mixing information between tokens. RMSNorm should normalize each token’s feature vector independently across `C`. In a causal model, mixing across time can also leak information from future positions.

</details>

## 6. Self-attention intuition

An embedding gives every occurrence of a token initially similar identity information. Self-attention allows each token representation to incorporate relevant information from preceding tokens.

Consider:

```text
The animal did not cross the street because it was tired.
```

When processing `it`, the model can assign attention to earlier tokens that help interpret the reference.

Each token produces three vectors:

- **Query:** what information am I looking for?
- **Key:** what information do I advertise?
- **Value:** what content should I contribute if selected?

The mathematical operation is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{D}} + M\right)V
$$

where:

- `D` is the head dimension.
- `M` is the causal mask.
- `QKᵀ` produces token-to-token compatibility scores.
- Softmax converts scores into weights.
- Multiplication by `V` computes weighted content.

## 7. Build single-head attention manually

Start with one sequence and small shapes:

```python
B, T, C = 1, 4, 8
x = torch.randn(B, T, C)

query_projection = nn.Linear(C, C, bias=False)
key_projection = nn.Linear(C, C, bias=False)
value_projection = nn.Linear(C, C, bias=False)

q = query_projection(x)
k = key_projection(x)
v = value_projection(x)

assert q.shape == (B, T, C)
assert k.shape == (B, T, C)
assert v.shape == (B, T, C)
```

Compare every query with every key:

```python
scores = q @ k.transpose(-2, -1)
assert scores.shape == (B, T, T)
```

Shape reasoning:

```text
q:   [B,T,C]
kᵀ:  [B,C,T]
out: [B,T,T]
```

Scale the scores:

```python
import math

scores = scores / math.sqrt(C)
```

Apply a causal mask:

```python
blocked = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1,
)

scores = scores.masked_fill(blocked, float("-inf"))
weights = F.softmax(scores, dim=-1)
```

The mask for `T=4` is:

```text
False True  True  True
False False True  True
False False False True
False False False False
```

So:

- Position 0 can attend only to position 0.
- Position 1 can attend to positions 0–1.
- Position 2 can attend to positions 0–2.
- Position 3 can attend to positions 0–3.

Confirm the attention distribution:

```python
torch.testing.assert_close(
    weights.sum(dim=-1),
    torch.ones(B, T),
)
assert weights[0, 0, 1:].max() == 0
assert weights[0, 1, 2:].max() == 0
```

Compute weighted values:

```python
output = weights @ v
assert output.shape == (B, T, C)
```

For query position `t`, the operation is:

$$
o_t = \sum_{j=0}^{t} \alpha_{t,j}v_j
$$

where the weights `α` are non-negative and sum to one.

### Checkpoint 4 — First token

In causal self-attention, what should the first token’s attention weights be?

<details>
<summary>Solution</summary>

The first token can attend only to itself. Its row must be:

```text
[1,0,0,...,0]
```

There may be any finite self-score before softmax, but every future score is `-inf`; softmax therefore assigns probability one to the only allowed position.

</details>

## 8. Why divide by the square root of head dimension?

If query and key elements have roughly zero mean and unit variance, a dot product over `D` components has variance that grows approximately with `D`.

Large-magnitude scores cause softmax to become nearly one-hot:

```python
scores = torch.tensor([1.0, 2.0, 3.0])

print(F.softmax(scores, dim=-1))
print(F.softmax(20 * scores, dim=-1))
```

The second distribution is far sharper. When softmax saturates, useful gradients can become very small.

Dividing by `sqrt(D)` approximately stabilizes the score scale as head dimension changes.

Test empirically:

```python
for D in [8, 32, 128, 512]:
    q = torch.randn(10_000, D)
    k = torch.randn(10_000, D)

    raw = (q * k).sum(dim=-1)
    scaled = raw / math.sqrt(D)

    print(
        D,
        "raw std =", round(raw.std().item(), 2),
        "scaled std =", round(scaled.std().item(), 2),
    )
```

You should observe raw standard deviation growing with `sqrt(D)` while scaled standard deviation remains around one.

## 9. Multi-head attention

One large attention operation can learn only one collection of query/key/value relationships per layer. Multi-head attention partitions the model dimension into multiple smaller heads.

```text
C = H × D
```

Example:

```text
model dimension C = 64
number of heads H = 4
head dimension D = 16
```

Shape transformation:

```text
[B,T,C]
→ [B,T,H,D]
→ [B,H,T,D]
```

Each head performs its own `[T,D] @ [D,T] → [T,T]` attention calculation.

## 10. Implement manual multi-head causal attention

```python
class ManualCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        number_of_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dimension % number_of_heads != 0:
            raise ValueError(
                "dimension must be divisible by number_of_heads"
            )

        self.dimension = dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = dimension // number_of_heads
        self.dropout = dropout

        # One matrix multiplication produces Q, K, and V together.
        self.qkv_projection = nn.Linear(
            dimension,
            3 * dimension,
            bias=False,
        )
        self.output_projection = nn.Linear(
            dimension,
            dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        if C != self.dimension:
            raise ValueError(
                f"Expected final dimension {self.dimension}, got {C}"
            )

        qkv = self.qkv_projection(x)  # [B,T,3C]
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                B,
                T,
                self.number_of_heads,
                self.head_dimension,
            ).transpose(1, 2)

        q = split_heads(q)  # [B,H,T,D]
        k = split_heads(k)  # [B,H,T,D]
        v = split_heads(v)  # [B,H,T,D]

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dimension)

        blocked = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(blocked, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(
            weights,
            p=self.dropout,
            training=self.training,
        )

        attended = weights @ v  # [B,H,T,D]

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(B, T, C)

        return self.output_projection(attended)
```

Test shapes and gradients:

```python
B, T, C, H = 2, 8, 64, 4

attention = ManualCausalSelfAttention(C, H).to(device)
x = torch.randn(B, T, C, device=device, requires_grad=True)
y = attention(x)

assert y.shape == x.shape

loss = y.square().mean()
loss.backward()

assert x.grad is not None
assert torch.isfinite(x.grad).all()

for parameter in attention.parameters():
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
```

### Parameter count

Ignoring bias:

```text
QKV projection: C × 3C = 3C²
output projection: C × C = C²
total: 4C²
```

The number of heads changes how channels are grouped, not the projection parameter count, as long as total `C` remains fixed.

### Checkpoint 5 — Trace every attention shape

For `B=2`, `T=16`, `C=96`, and `H=6`, write the shape after each operation:

1. `x`
2. combined `qkv`
3. `q` before splitting heads
4. `q` after splitting heads
5. `scores`
6. `weights @ v`
7. recombined heads
8. output projection

<details>
<summary>Solution</summary>

Head dimension is `D=C/H=16`.

```text
x:                    [2,16,96]
qkv:                  [2,16,288]
q before split:       [2,16,96]
q after split:        [2,6,16,16]
scores:               [2,6,16,16]
weights @ v:          [2,6,16,16]
recombined heads:     [2,16,96]
output projection:    [2,16,96]
```

</details>

## 11. Verify causal behavior, not just shapes

A correctly shaped implementation can still leak future information.

Create two inputs with identical prefixes but different suffixes:

```python
torch.manual_seed(42)

B, T, C, H = 1, 8, 32, 4
attention = ManualCausalSelfAttention(C, H, dropout=0.0).to(device)
attention.eval()

original = torch.randn(B, T, C, device=device)
modified = original.clone()

prefix_length = 4
modified[:, prefix_length:] = torch.randn_like(
    modified[:, prefix_length:]
) * 100

with torch.inference_mode():
    original_output = attention(original)
    modified_output = attention(modified)

torch.testing.assert_close(
    original_output[:, :prefix_length],
    modified_output[:, :prefix_length],
    atol=1e-5,
    rtol=1e-5,
)
```

Changing future tokens must not affect outputs for earlier positions. Outputs at or after `prefix_length` may change.

This causality test is more meaningful than simply checking that a mask tensor exists.

## 12. Use PyTorch scaled dot-product attention

The manual implementation is ideal for learning but creates an explicit `[B,H,T,T]` score tensor. Optimized attention kernels can reduce memory traffic and avoid materializing all intermediates in the same way.

PyTorch provides:

```python
F.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
)
```

Replace the manual score, mask, softmax, and value multiplication with:

```python
attended = F.scaled_dot_product_attention(
    q,
    k,
    v,
    dropout_p=self.dropout if self.training else 0.0,
    is_causal=True,
)
```

Important: scaled dot-product attention applies dropout according to the `dropout_p` argument. It does not automatically inspect the surrounding module’s training flag. Pass `0.0` during evaluation.

Build an optimized version:

```python
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        number_of_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dimension % number_of_heads != 0:
            raise ValueError(
                "dimension must be divisible by number_of_heads"
            )

        self.dimension = dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = dimension // number_of_heads
        self.dropout = dropout

        self.qkv_projection = nn.Linear(
            dimension,
            3 * dimension,
            bias=False,
        )
        self.output_projection = nn.Linear(
            dimension,
            dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv_projection(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                B,
                T,
                self.number_of_heads,
                self.head_dimension,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(B, T, C)

        return self.output_projection(attended)
```

## 13. Verify manual attention against SDPA

To compare implementations, give them identical weights and disable dropout:

```python
torch.manual_seed(42)

B, T, C, H = 2, 8, 64, 4
x = torch.randn(B, T, C, device=device)

manual = ManualCausalSelfAttention(C, H, dropout=0.0).to(device)
optimized = CausalSelfAttention(C, H, dropout=0.0).to(device)

optimized.load_state_dict(manual.state_dict())
manual.eval()
optimized.eval()

with torch.inference_mode():
    manual_output = manual(x)
    optimized_output = optimized(x)

torch.testing.assert_close(
    manual_output,
    optimized_output,
    atol=1e-5,
    rtol=1e-4,
)
```

Floating-point operation order may differ across CPU, MPS, and CUDA kernels. Small numerical differences do not necessarily mean one result is wrong.

## 14. Why attention needs positional information

Attention scores are based on token content. Without positional information, the operation does not inherently know whether a key came from position 2 or position 200.

Day 2 showed learned position embeddings. Today we use rotary position embeddings (RoPE), a common choice in decoder-only LLMs.

RoPE rotates pairs of query and key features by position-dependent angles. It does not add a separate position vector to the residual stream.

For one pair `(x_even,x_odd)` and angle `θ`:

$$
\begin{bmatrix}
x'_{even} \\
x'_{odd}
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x_{even} \\
x_{odd}
\end{bmatrix}
$$

Different feature pairs rotate at different frequencies. Position changes the angle.

Important properties:

- RoPE is applied to queries and keys, not values.
- Rotation preserves each pair’s Euclidean norm.
- Query-key dot products become sensitive to relative position.
- Head dimension must be even for this paired implementation.

## 15. Implement RoPE

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs: (a, b) -> (-b, a)."""
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10_000.0,
) -> torch.Tensor:
    """
    Apply rotary position embeddings.

    x shape:         [B,H,T,D]
    positions shape: [T]
    """
    head_dimension = x.size(-1)

    if head_dimension % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    inverse_frequencies = 1.0 / (
        base
        ** (
            torch.arange(
                0,
                head_dimension,
                2,
                device=x.device,
                dtype=torch.float32,
            )
            / head_dimension
        )
    )

    # [T] outer [D/2] -> [T,D/2]
    angles = torch.outer(
        positions.to(torch.float32),
        inverse_frequencies,
    )

    # Repeat each pair angle for its even and odd features: [T,D]
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)

    cosine = angles.cos()[None, None].to(dtype=x.dtype)
    sine = angles.sin()[None, None].to(dtype=x.dtype)

    return x * cosine + rotate_half(x) * sine
```

Test shapes and norm preservation:

```python
B, H, T, D = 2, 4, 16, 8
x = torch.randn(B, H, T, D)
positions = torch.arange(T)

rotated = apply_rope(x, positions)

assert rotated.shape == x.shape

torch.testing.assert_close(
    rotated.float().norm(dim=-1),
    x.float().norm(dim=-1),
    atol=1e-5,
    rtol=1e-5,
)
```

Position zero has angle zero at every frequency, so it remains unchanged:

```python
torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0])
```

### Checkpoint 6 — Understand the rotation

For vector pair `(3,4)`:

1. What does `rotate_half` produce?
2. What happens at angle zero?
3. Why does the vector length remain five after any rotation?

<details>
<summary>Solution</summary>

1. `rotate_half([3,4])` produces `[-4,3]`.
2. At angle zero, `cos(0)=1` and `sin(0)=0`, so the original vector remains `[3,4]`.
3. A rotation changes direction but not length. The length remains `sqrt(3²+4²)=5`.

</details>

## 16. Add RoPE to attention

Apply it after splitting heads and before computing query-key scores:

```python
class RotaryCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        number_of_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10_000.0,
    ):
        super().__init__()

        if dimension % number_of_heads != 0:
            raise ValueError(
                "dimension must be divisible by number_of_heads"
            )

        self.dimension = dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = dimension // number_of_heads
        self.dropout = dropout
        self.rope_base = rope_base

        if self.head_dimension % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")

        self.qkv_projection = nn.Linear(
            dimension,
            3 * dimension,
            bias=False,
        )
        self.output_projection = nn.Linear(
            dimension,
            dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_projection(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                B,
                T,
                self.number_of_heads,
                self.head_dimension,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        positions = torch.arange(T, device=x.device)
        q = apply_rope(q, positions, base=self.rope_base)
        k = apply_rope(k, positions, base=self.rope_base)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.reshape(B, T, C)

        return self.output_projection(attended)
```

Later, KV-cached decoding will pass positions beginning at the current cache length rather than always beginning at zero.

## 17. The feed-forward network

Attention mixes information across token positions. The feed-forward network transforms each token independently across channels.

```text
attention: mixes across T
feed-forward: transforms across C independently at each B,T location
```

A standard two-layer MLP expands and contracts the channel dimension:

```text
[B,T,C]
→ [B,T,F]
→ activation
→ [B,T,C]
```

`F` is the hidden or intermediate dimension and is usually larger than `C`.

## 18. SiLU and SwiGLU

SiLU is:

$$
\operatorname{SiLU}(x)=x\,\sigma(x)
$$

```python
values = torch.linspace(-5, 5, steps=11)
print(F.silu(values))
```

SwiGLU uses two input projections:

$$
\operatorname{SwiGLU}(x)
=
W_{down}\left(
\operatorname{SiLU}(W_{gate}x)
\odot
(W_{up}x)
\right)
$$

One branch acts as a learned gate over the other branch.

## 19. Implement SwiGLU

```python
class SwiGLU(nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gate_projection = nn.Linear(
            dimension,
            hidden_dimension,
            bias=False,
        )
        self.up_projection = nn.Linear(
            dimension,
            hidden_dimension,
            bias=False,
        )
        self.down_projection = nn.Linear(
            hidden_dimension,
            dimension,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_projection(x))
        values = self.up_projection(x)
        hidden = gate * values
        return self.dropout(self.down_projection(hidden))
```

Test:

```python
B, T, C, FFD = 2, 8, 64, 192

feed_forward = SwiGLU(C, FFD).to(device)
x = torch.randn(B, T, C, device=device)
y = feed_forward(x)

assert y.shape == x.shape
```

Shape trace:

```text
x:                 [B,T,C]
gate projection:   [B,T,FFD]
up projection:     [B,T,FFD]
elementwise gate:  [B,T,FFD]
down projection:   [B,T,C]
```

Parameter count without biases:

```text
gate: C × FFD
up:   C × FFD
down: FFD × C
total: 3 × C × FFD
```

### Checkpoint 7 — Attention versus feed-forward

Which component can directly mix information between token positions?

<details>
<summary>Solution</summary>

Attention directly mixes information across sequence positions through the `[T,T]` attention matrix. SwiGLU applies the same channel transformation independently to every `[b,t]` token vector. Context already collected by attention can be transformed by SwiGLU, but SwiGLU itself does not move information from one position to another.

</details>

## 20. Residual connections

A residual connection adds a sublayer’s output back to its input:

$$
y=x+f(x)
$$

```python
updated = x + attention_output
```

Why residuals help:

- A sublayer can learn a correction rather than reconstructing everything.
- Information can flow through an identity path.
- Gradients have a direct route through addition.
- Stacking many blocks becomes more stable.

Shape equality is mandatory:

```text
x:             [B,T,C]
attention(x):  [B,T,C]
x + attention: [B,T,C]
```

This is why attention’s output projection and SwiGLU’s down projection return to dimension `C`.

## 21. Pre-normalization

In a pre-norm transformer, normalization occurs before each sublayer:

```python
x = x + attention(norm1(x))
x = x + feed_forward(norm2(x))
```

The residual stream itself has a direct identity path around both computations.

This differs from a post-norm arrangement:

```python
x = norm1(x + attention(x))
x = norm2(x + feed_forward(x))
```

We will use pre-norm, which is common in modern decoder models and behaves well when stacking many layers.

## 22. Assemble one complete transformer block

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        number_of_heads: int,
        hidden_dimension: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        rope_base: float = 10_000.0,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(dimension, eps=norm_eps)
        self.attention = RotaryCausalSelfAttention(
            dimension=dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
            rope_base=rope_base,
        )

        self.feed_forward_norm = RMSNorm(dimension, eps=norm_eps)
        self.feed_forward = SwiGLU(
            dimension=dimension,
            hidden_dimension=hidden_dimension,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x
```

Trace it:

```text
x [B,T,C]
│
├─ attention_norm(x) [B,T,C]
├─ attention(...) [B,T,C]
└─ residual add [B,T,C]
│
├─ feed_forward_norm(x) [B,T,C]
├─ feed_forward(...) [B,T,C]
└─ residual add [B,T,C]
```

## 23. Full block smoke test

```python
torch.manual_seed(42)

B = 2
T = 16
C = 64
H = 4
FFD = 192

block = TransformerBlock(
    dimension=C,
    number_of_heads=H,
    hidden_dimension=FFD,
    dropout=0.0,
).to(device)

x = torch.randn(
    B,
    T,
    C,
    device=device,
    requires_grad=True,
)

y = block(x)

assert y.shape == x.shape
assert torch.isfinite(y).all()

loss = y.square().mean()
loss.backward()

assert x.grad is not None
assert torch.isfinite(x.grad).all()

for name, parameter in block.named_parameters():
    assert parameter.grad is not None, f"Missing gradient: {name}"
    assert torch.isfinite(parameter.grad).all(), (
        f"Non-finite gradient: {name}"
    )

print("output shape:", y.shape)
print("loss:", loss.item())
print(
    "parameters:",
    sum(parameter.numel() for parameter in block.parameters()),
)
```

## 24. Causality test for the complete block

```python
torch.manual_seed(42)

block.eval()
original = torch.randn(1, 12, C, device=device)
modified = original.clone()

prefix_length = 6
modified[:, prefix_length:] = torch.randn_like(
    modified[:, prefix_length:]
) * 100

with torch.inference_mode():
    output_original = block(original)
    output_modified = block(modified)

torch.testing.assert_close(
    output_original[:, :prefix_length],
    output_modified[:, :prefix_length],
    atol=1e-5,
    rtol=1e-4,
)
```

Why this works:

- RMSNorm operates independently per token.
- Causal attention prevents access to future positions.
- SwiGLU operates independently per token.
- Residual addition operates position by position.

If this test fails with dropout disabled, the block contains future-token leakage or an operation that mixes the time dimension incorrectly.

## 25. Stack multiple blocks

A full transformer repeats the same block structure with different learned weights:

```python
class TransformerStack(nn.Module):
    def __init__(
        self,
        number_of_layers: int,
        dimension: int,
        number_of_heads: int,
        hidden_dimension: int,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dimension=dimension,
                number_of_heads=number_of_heads,
                hidden_dimension=hidden_dimension,
            )
            for _ in range(number_of_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
```

Use `nn.ModuleList`, not a plain Python list. `ModuleList` registers child modules so parameters appear in `state_dict`, move with `.to(device)`, and are visible to optimizers.

```python
stack = TransformerStack(
    number_of_layers=4,
    dimension=64,
    number_of_heads=4,
    hidden_dimension=192,
).to(device)

x = torch.randn(2, 16, 64, device=device)
y = stack(x)

assert y.shape == x.shape
```

Day 4 will place token embeddings before this stack, add final normalization and a vocabulary output head, and train the resulting language model.

## 26. Parameter accounting

For one block without biases:

### Attention

```text
QKV projection: 3C²
output projection: C²
attention total: 4C²
```

### SwiGLU

```text
gate projection: C × FFD
up projection:   C × FFD
down projection: FFD × C
SwiGLU total: 3C × FFD
```

### RMSNorm

```text
two learned scale vectors: 2C
```

### Total

```text
4C² + 3C×FFD + 2C
```

Verify against PyTorch:

```python
C, FFD = 64, 192
expected = 4 * C**2 + 3 * C * FFD + 2 * C
actual = sum(p.numel() for p in block.parameters())

print("expected:", expected)
print("actual:  ", actual)
assert expected == actual
```

Parameter accounting is an essential research-engineering habit. It catches architectural mistakes and helps estimate model memory.

At two bytes per parameter in BF16 or FP16, parameter storage alone is approximately:

```text
parameter_count × 2 bytes
```

Training requires additional memory for gradients, optimizer state, activations, and temporary buffers.

## 27. Common mistakes

### Scaling by model dimension instead of head dimension

Wrong:

```python
scores = scores / math.sqrt(C)
```

Correct after heads are split:

```python
scores = scores / math.sqrt(D)
```

Each dot product contains `D` terms.

### Softmax over the wrong dimension

For scores `[B,H,T_query,T_key]`, normalize over keys:

```python
weights = F.softmax(scores, dim=-1)
```

### Reversing mask meaning

In the manual implementation, `True` means blocked because it is passed to `masked_fill`. Some PyTorch APIs use `True` to mean allowed. Always verify the specific API and add a causality test.

### Forgetting `.contiguous()` before `view`

After transposing heads:

```python
attended = attended.transpose(1, 2).contiguous().view(B, T, C)
```

Using `reshape` is also acceptable and may create a copy when necessary.

### Applying RoPE to values

For this architecture, rotate `q` and `k`, not `v`.

### Using an odd head dimension

The paired RoPE implementation requires `D` to be even.

### Normalizing across sequence length

RMSNorm reduces across final channel dimension `C`, not token dimension `T`.

### Forgetting residual connections

These are not equivalent:

```python
x = attention(norm(x))      # loses identity path
x = x + attention(norm(x))  # residual block
```

### Creating sublayers in `forward`

Create trainable layers once in `__init__` so their parameters are registered and persistent.

### SDPA dropout during evaluation

Pass:

```python
dropout_p=self.dropout if self.training else 0.0
```

### Testing only output shapes

Shape tests do not detect future-token leakage, incorrect mask direction, missing gradients, NaNs, or differences from a trusted reference.

## 28. Hands-on exercises

### Exercise 1 — Write RMSNorm without looking

Requirements:

- One learned scale per channel.
- Reduction over the final dimension.
- `keepdim=True`.
- Float32 calculation for the mean square.
- Output dtype matches input dtype.

Tests:

```python
norm = RMSNorm(16)
x = torch.randn(2, 5, 16)
y = norm(x)

assert y.shape == x.shape
assert y.dtype == x.dtype
assert sum(p.numel() for p in norm.parameters()) == 16
```

### Exercise 2 — Implement single-head causal attention

Do not use `F.scaled_dot_product_attention`. Your implementation must create scores, scale them, mask future positions, apply softmax, and combine values.

Test that the first attention row is `[1,0,...,0]`.

### Exercise 3 — Convert single-head to multi-head

Given input `[B,T,C]`, produce:

```text
q, k, v: [B,H,T,D]
scores:   [B,H,T,T]
output:   [B,T,C]
```

Add assertions after every transformation.

### Exercise 4 — Prove causality

Create two inputs sharing the first half but with different second halves. Assert their first-half outputs match.

### Exercise 5 — Compare manual attention and SDPA

Give both modules identical parameters, disable dropout, and use `torch.testing.assert_close`.

### Exercise 6 — Test RoPE invariants

Verify:

- Output shape equals input shape.
- Position zero is unchanged.
- Rotation preserves vector norms.
- Odd head dimension raises `ValueError`.

### Exercise 7 — Parameter count

For `C=512`, `H=8`, and `FFD=1536`, calculate one block’s parameter count before running Python. Then verify it programmatically.

<details>
<summary>Solution</summary>

```text
attention = 4 × 512² = 1,048,576
SwiGLU    = 3 × 512 × 1536 = 2,359,296
RMSNorm   = 2 × 512 = 1,024
total     = 3,408,896
```

The head count does not change this total while `C` remains fixed.

</details>

### Exercise 8 — Full backward pass

Run a scalar loss backward through the complete block and verify every trainable parameter has a finite gradient.

## 29. Final integrated challenge

Create `day03_exercises.py` containing:

1. Device selection and a fixed random seed.
2. Your `RMSNorm` implementation.
3. Manual causal multi-head attention.
4. Causality tests for manual attention.
5. SDPA-based causal attention.
6. A numerical comparison between manual and SDPA attention.
7. `rotate_half` and `apply_rope`.
8. RoPE invariant tests.
9. Rotary causal self-attention.
10. `SwiGLU`.
11. A pre-norm residual `TransformerBlock`.
12. Shape, finiteness, causality, and gradient tests.
13. A printed parameter breakdown.

Expected final output should resemble:

```text
device: mps
RMSNorm tests: passed
manual attention shapes: passed
manual attention causality: passed
manual vs SDPA: passed
RoPE invariants: passed
SwiGLU shapes: passed
transformer block shapes: passed
transformer block causality: passed
all gradients finite: passed
parameter count: 53,376
```

The exact parameter count depends on your chosen dimensions.

## 30. Completion checklist

Day 3 is complete when you can explain:

- Why transformer blocks consume and return `[B,T,C]`.
- Why trainable layers belong in `__init__`.
- Why a linear weight is stored as `[C_out,C_in]`.
- Why RMSNorm reduces over `C` independently for every token.
- Why attention produces a `[T,T]` relationship matrix per head.
- Why attention scores are divided by `sqrt(D)`.
- Why the causal mask blocks the upper triangle.
- Why softmax operates over the key dimension.
- Why `C` must be divisible by the number of heads.
- How `[B,T,C]` becomes `[B,H,T,D]` and returns to `[B,T,C]`.
- Why causal correctness requires changing future inputs and testing past outputs.
- Why RoPE is applied to queries and keys.
- Why rotation preserves vector magnitude.
- How attention and SwiGLU perform different jobs.
- Why residual connections require matching shapes.
- What pre-normalization means.
- Why SDPA dropout must be explicitly disabled during evaluation.
- How to calculate the block’s parameter count.

## References

- [PyTorch `nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch `nn.RMSNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [PyTorch SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [PyTorch functional API, including SiLU](https://docs.pytorch.org/docs/stable/nn.functional)
- [PyTorch `ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList)
