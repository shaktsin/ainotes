# Day 1 — PyTorch Tensors, Shapes, and Autograd

**Status:** Completed

Today’s objective is to understand the object flowing through every neural network: the tensor.

By the end of this lesson, you should be able to:

- Read and manipulate tensor shapes confidently.
- Explain batch, sequence, channel, head, and vocabulary dimensions.
- Distinguish elementwise multiplication from matrix multiplication.
- Understand broadcasting, transposition, and strides.
- Construct causal self-attention using basic tensor operations.
- Explain how PyTorch builds a computation graph.
- Calculate gradients using `.backward()`.
- Recognize common tensor and autograd mistakes.

Run every example interactively. Change values and deliberately break things—the error messages are part of the lesson.

## 1. What is a tensor?

A tensor is an n-dimensional rectangular collection of numbers.

| Dimensions | Mathematical name | Example shape |
|---:|---|---|
| 0 | Scalar | `[]` |
| 1 | Vector | `[4]` |
| 2 | Matrix | `[3,4]` |
| 3 | Stack of matrices | `[2,3,4]` |
| 4 | Batch of multi-head sequences | `[B,H,T,D]` |

```python
import torch

scalar = torch.tensor(7.0)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
three_dimensional = torch.tensor([
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]],
])

for name, tensor in [
    ("scalar", scalar),
    ("vector", vector),
    ("matrix", matrix),
    ("3D", three_dimensional),
]:
    print(
        name,
        "shape =", tensor.shape,
        "dimensions =", tensor.ndim,
        "elements =", tensor.numel(),
    )
```

Expected shapes:

```text
scalar: []
vector: [3]
matrix: [2,3]
3D:     [2,2,2]
```

A tensor contains more than its visible numbers:

```text
Tensor
├── data
├── shape
├── dtype
├── device
├── strides
└── gradient information
```

## 2. Understanding shapes

In a language model, activations are commonly arranged as `[B,T,C]`:

```python
B = 2   # batch size: sequences processed together
T = 3   # sequence length: tokens in each sequence
C = 4   # channels: numbers representing each token

x = torch.randn(B, T, C)
print(x.shape)  # torch.Size([2, 3, 4])
```

Visualize it as:

```text
Batch 0
├── Token 0 → 4 numbers
├── Token 1 → 4 numbers
└── Token 2 → 4 numbers

Batch 1
├── Token 0 → 4 numbers
├── Token 1 → 4 numbers
└── Token 2 → 4 numbers
```

Indexing with an integer removes that dimension:

```python
assert x[0].shape == (T, C)
assert x[0, 1].shape == (C,)
assert x[0, 1, 2].shape == ()

print(x[0].shape)    # [3,4], batch dimension removed
print(x[0:1].shape)  # [1,3,4], batch dimension preserved
```

This distinction matters because many model functions require a batch dimension even when processing one example.

## 3. Dtypes

A tensor’s `dtype` specifies how every number is represented.

```python
integers = torch.tensor([1, 2, 3])
floats = torch.tensor([1.0, 2.0, 3.0])
small_floats = floats.to(torch.float16)

print(integers.dtype)     # torch.int64
print(floats.dtype)       # torch.float32
print(small_floats.dtype) # torch.float16
```

Typical language-model dtypes:

| Dtype | Typical use |
|---|---|
| `torch.int64` / `torch.long` | Token IDs and indices |
| `torch.float32` | Stable training and reference calculations |
| `torch.float16` | Reduced-memory GPU computation |
| `torch.bfloat16` | Reduced-memory training and inference |
| `torch.int8` | Quantized inference |

Token IDs are integers, while model activations are floating-point:

```python
token_ids = torch.tensor([[10, 25, 91, 7]], dtype=torch.long)
embedding = torch.nn.Embedding(num_embeddings=256, embedding_dim=64)
activations = embedding(token_ids)

print(token_ids.shape, token_ids.dtype)   # [1,4], integer
print(activations.shape, activations.dtype)  # [1,4,64], float
```

An embedding uses each integer token ID to select a learned floating-point row from its weight matrix.

## 4. Devices

A tensor exists on a CPU, GPU, or another accelerator.

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using:", device)
```

Move tensors and models using `.to(device)`:

```python
x = torch.randn(2, 4).to(device)
model = torch.nn.Linear(4, 8).to(device)
output = model(x)
```

Operations require participating tensors to be on the same device:

```python
x = torch.randn(2, 3, device=device)
y = torch.randn(2, 3).to(x.device)
z = x + y
```

## 5. Creating tensors

```python
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random_normal = torch.randn(2, 3)
random_uniform = torch.rand(2, 3)
integer_range = torch.arange(0, 12)
evenly_spaced = torch.linspace(0, 1, steps=5)
```

- `torch.rand` samples approximately uniformly between 0 and 1.
- `torch.randn` samples from a normal distribution with mean 0 and standard deviation 1.

Create tensors matching an existing tensor’s shape, dtype, and device:

```python
x = torch.randn(2, 3, device=device)
a = torch.zeros_like(x)
b = torch.ones_like(x)
c = torch.randn_like(x)
```

## 6. Reshaping tensors

Reshaping changes the logical grouping of elements without changing their count.

```python
x = torch.arange(24)
x = x.reshape(2, 3, 4)

assert x.numel() == 24
assert x.shape == (2, 3, 4)
```

The product of the new dimensions must remain 24. PyTorch can infer one dimension:

```python
x = torch.arange(24)
a = x.reshape(2, -1)     # [2,12]
b = x.reshape(2, 3, -1)  # [2,3,4]
```

Add dimensions with `unsqueeze`:

```python
x = torch.randn(3, 4)

print(x.unsqueeze(0).shape)   # [1,3,4]
print(x.unsqueeze(1).shape)   # [3,1,4]
print(x.unsqueeze(-1).shape)  # [3,4,1]
```

Remove a size-one dimension with `squeeze`:

```python
x = torch.randn(1, 3, 1, 4)
y = x.squeeze(2)
print(y.shape)  # [1,3,4]
```

Prefer specifying the dimension. Unrestricted `squeeze()` can accidentally remove a batch dimension when batch size is one.

## 7. Transpose and permute

`reshape` regroups dimensions; `transpose` and `permute` reorder dimensions.

```python
x = torch.randn(2, 3, 4)

y = x.transpose(1, 2)
print(y.shape)  # [2,4,3]

z = x.permute(2, 0, 1)
print(z.shape)  # [4,2,3]
```

Attention commonly changes:

```text
[batch, sequence, heads, head_dimension]
→ [batch, heads, sequence, head_dimension]
```

```python
B, T, H, D = 2, 8, 4, 16
x = torch.randn(B, T, H, D).transpose(1, 2)
assert x.shape == (B, H, T, D)
```

## 8. Strides and contiguous memory

A shape describes logical organization. Strides describe how to move through the underlying memory.

```python
x = torch.arange(12).reshape(3, 4)

print(x.shape)           # [3,4]
print(x.stride())        # usually [4,1]
print(x.is_contiguous()) # True
```

To move one row forward, skip four elements. To move one column forward, skip one.

Transposition normally changes the view rather than moving every value:

```python
y = x.transpose(0, 1)

print(y.shape)           # [4,3]
print(y.stride())        # usually [1,4]
print(y.is_contiguous()) # False
```

`view` requires a compatible memory layout:

```python
flattened = y.contiguous().view(-1)
```

`reshape` may create a contiguous copy when necessary:

```python
flattened = y.reshape(-1)
```

Practical rule:

- Use `reshape` when you primarily care about the target shape.
- Use `view` when you understand the memory layout.
- After `transpose`, expect that `contiguous()` may be necessary.

## 9. Elementwise operations

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])

print(a + b)  # [11,22,33]
print(a * b)  # [10,40,90]
print(a**2)   # [1,4,9]
```

For matrices, `*` still means elementwise multiplication. It is not matrix multiplication.

## 10. Matrix multiplication

Matrix multiplication combines rows of the first matrix with columns of the second:

```python
a = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])
b = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0],
])

print(a @ b)
```

Result:

```text
[[19,22],
 [43,50]]
```

The shape rule is:

```text
[M,K] @ [K,N] → [M,N]
```

For batched neural-network inputs:

```python
B, T, C, OUT = 2, 3, 4, 8
x = torch.randn(B, T, C)
weight = torch.randn(C, OUT)
y = x @ weight

assert y.shape == (B, T, OUT)
```

PyTorch performs multiplication over the final relevant dimensions and preserves preceding batch dimensions.

## 11. Broadcasting

Broadcasting combines compatible shapes without manually copying data.

```python
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
bias = torch.tensor([10.0, 20.0, 30.0])

print(x + bias)
```

Shapes `[2,3]` and `[3]` are compatible. Conceptually, `bias` is repeated across both rows.

Compare dimensions from right to left. Dimensions are compatible when they are equal, one equals `1`, or one tensor lacks that dimension:

```text
[2,3,4] +     [4] → valid
[2,3,4] +   [1,4] → valid
[2,3,4] + [2,1,4] → valid
[2,3,4] +   [2,4] → invalid
```

`keepdim=True` preserves a reduced dimension for broadcasting:

```python
x = torch.randn(2, 3, 4)
mean = x.mean(dim=-1, keepdim=True)  # [2,3,1]
centered = x - mean                  # [2,3,4]
```

This pattern is fundamental to normalization layers.

## 12. Reductions

A reduction combines multiple values into fewer values:

```python
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print(x.sum())       # 21
print(x.sum(dim=0))  # [5,7,9]
print(x.sum(dim=1))  # [6,15]
```

Think of `dim` as the axis being eliminated. For `[B,T,C]`:

```python
x.mean(dim=-1).shape       # [B,T]
x.mean(dim=1).shape        # [B,C]
x.sum(dim=(1, 2)).shape    # [B]
```

## 13. Softmax

Softmax converts logits into probabilities:

$$
\operatorname{softmax}(z_i)
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

```python
logits = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 2.0, 2.0],
])
probabilities = torch.softmax(logits, dim=-1)

print(probabilities)
print(probabilities.sum(dim=-1))  # [1,1]
```

For attention scores `[B,H,T_query,T_key]`, softmax must operate over the key dimension:

```python
attention_probabilities = torch.softmax(scores, dim=-1)
```

Using the wrong dimension can produce valid-looking numbers without raising an error, making it a dangerous semantic bug.

## 14. Self-attention step by step

Use small dimensions:

```python
B = 1   # sequences
T = 3   # tokens per sequence
C = 4   # model dimension
H = 2   # attention heads
D = 2   # dimensions per head

assert C == H * D
x = torch.randn(B, T, C)
```

### 14.1 Produce queries, keys, and values

```python
wq = torch.randn(C, C)
wk = torch.randn(C, C)
wv = torch.randn(C, C)

q = x @ wq
k = x @ wk
v = x @ wv
```

Useful intuition:

- Query: what information is this token looking for?
- Key: what information does this token offer?
- Value: what content should be returned if selected?

### 14.2 Split channels into heads

```python
q = q.reshape(B, T, H, D).transpose(1, 2)
k = k.reshape(B, T, H, D).transpose(1, 2)
v = v.reshape(B, T, H, D).transpose(1, 2)

assert q.shape == (B, H, T, D)
```

### 14.3 Compare queries with keys

```python
scores = q @ k.transpose(-2, -1)
assert scores.shape == (B, H, T, T)
```

Shape reasoning:

```text
[B,H,T,D] @ [B,H,D,T] → [B,H,T,T]
```

Each `T×T` matrix contains one compatibility score for every query-key pair.

### 14.4 Scale scores

```python
import math

scores = scores / math.sqrt(D)
```

Dot products grow with vector dimension. Scaling controls their magnitude and prevents softmax from becoming excessively sharp.

### 14.5 Apply causal masking

A token must not see future tokens during next-token prediction:

```python
future_mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1,
)
scores = scores.masked_fill(future_mask, float("-inf"))
```

### 14.6 Convert scores to probabilities

```python
attention = torch.softmax(scores, dim=-1)

assert torch.allclose(
    attention.sum(dim=-1),
    torch.ones(B, H, T),
)
assert attention[..., 0, 1:].max() == 0
assert attention[..., 1, 2:].max() == 0
```

### 14.7 Compute weighted values

```python
output = attention @ v
assert output.shape == (B, H, T, D)
```

Each output token is now a weighted combination of value vectors.

### 14.8 Recombine heads

```python
output = output.transpose(1, 2).contiguous().reshape(B, T, C)
assert output.shape == (B, T, C)
```

Complete shape flow:

```text
[B,T,C]
→ Q, K, V: [B,T,C]
→ split heads: [B,H,T,D]
→ QKᵀ: [B,H,T,T]
→ mask and softmax
→ weighted V: [B,H,T,D]
→ combine heads: [B,T,C]
```

## 15. What is autograd?

Training asks: how should every model parameter change to reduce the loss?

A gradient measures how a small change in one value affects another. For:

$$
y=x^2
$$

the derivative is:

$$
\frac{dy}{dx}=2x
$$

At `x=3`, the derivative is 6. A small change in `x` causes approximately six times that change in `y`.

PyTorch records operations involving tensors with `requires_grad=True` and applies the chain rule automatically.

## 16. Scalar autograd

```python
x = torch.tensor(3.0, requires_grad=True)
y = x**2

print(y.grad_fn)

y.backward()
print(x.grad)  # 6
```

`grad_fn` records how `y` was produced. `backward()` traverses this computation graph from the scalar output toward its inputs.

## 17. The chain rule

For:

$$
a=x^2,\qquad b=3a,\qquad L=b+1
$$

we have:

$$
L=3x^2+1,\qquad \frac{dL}{dx}=6x
$$

At `x=2`, the gradient is 12:

```python
x = torch.tensor(2.0, requires_grad=True)
a = x**2
b = 3 * a
loss = b + 1

loss.backward()
print(x.grad)  # 12
```

Autograd combines local derivatives:

$$
\frac{dL}{dx}
=
\frac{dL}{db}
\frac{db}{da}
\frac{da}{dx}
$$

This repeated chain-rule application is backpropagation.

## 18. Vector gradients

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x**2).sum()
loss.backward()

print(x.grad)  # [2,4,6]
```

Here:

$$
L=x_1^2+x_2^2+x_3^2,
\qquad
\nabla_xL=[2x_1,2x_2,2x_3]
$$

A training loss is normally scalar, giving backpropagation one clear starting point.

## 19. Gradients accumulate

```python
x = torch.tensor(2.0, requires_grad=True)

(x**2).backward()
print(x.grad)  # 4

(x**2).backward()
print(x.grad)  # 8
```

The second backward pass adds another 4. It does not replace the existing gradient.

Reset model gradients before the next ordinary training iteration:

```python
optimizer.zero_grad(set_to_none=True)
```

For an individual tensor:

```python
x.grad = None
```

## 20. Leaf and non-leaf tensors

A leaf tensor was created directly rather than produced by another tracked operation:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y**2

print(x.is_leaf)  # True
print(y.is_leaf)  # False
print(z.is_leaf)  # False
```

Gradients are retained for leaf tensors by default. Retain an intermediate gradient explicitly when debugging:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
y.retain_grad()

z = y**2
z.backward()

print(x.grad)
print(y.grad)
```

Neural-network `nn.Parameter` objects are normally leaf tensors.

## 21. A tiny trainable model

Learn the relationship `y = 2x + 1` from examples:

```python
x = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([3.0, 5.0, 7.0])

weight = torch.tensor(0.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

for step in range(100):
    prediction = weight * x + bias
    loss = ((prediction - target)**2).mean()

    loss.backward()

    with torch.no_grad():
        weight -= 0.1 * weight.grad
        bias -= 0.1 * bias.grad

    weight.grad = None
    bias.grad = None

    if step % 10 == 0:
        print(
            step,
            "loss =", round(loss.item(), 6),
            "weight =", round(weight.item(), 3),
            "bias =", round(bias.item(), 3),
        )
```

The parameters should approach `weight ≈ 2` and `bias ≈ 1`.

```text
parameters
→ predictions
→ loss
→ gradients
→ parameter update
→ repeat
```

Every language-model training loop is a much larger version of this process.

## 22. `no_grad`, `inference_mode`, and `detach`

`torch.no_grad()` temporarily disables gradient recording. Use it during validation or manual parameter updates:

```python
with torch.no_grad():
    prediction = weight * x + bias
```

`torch.inference_mode()` is a stronger inference-only mode. Use it for generation and deployment when backward will not be called:

```python
with torch.inference_mode():
    prediction = weight * x + bias
```

`detach()` returns a tensor disconnected from the current graph:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2
detached_y = y.detach()

print(y.requires_grad)           # True
print(detached_y.requires_grad)  # False
```

This is useful when logging or storing results without retaining their entire computation graph.

## 23. Autograd through attention

```python
B, T, C, H = 2, 4, 8, 2
D = C // H

x = torch.randn(B, T, C, requires_grad=True)
wq = torch.randn(C, C, requires_grad=True)
wk = torch.randn(C, C, requires_grad=True)
wv = torch.randn(C, C, requires_grad=True)

q = (x @ wq).reshape(B, T, H, D).transpose(1, 2)
k = (x @ wk).reshape(B, T, H, D).transpose(1, 2)
v = (x @ wv).reshape(B, T, H, D).transpose(1, 2)

scores = q @ k.transpose(-2, -1) / D**0.5
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float("-inf"))
probabilities = scores.softmax(dim=-1)
output = probabilities @ v

loss = output.square().mean()
loss.backward()
```

Inspect gradients:

```python
for name, parameter in [
    ("x", x),
    ("wq", wq),
    ("wk", wk),
    ("wv", wv),
]:
    print(
        name,
        "shape =", parameter.grad.shape,
        "norm =", parameter.grad.norm().item(),
    )
```

PyTorch differentiated through the full graph:

```text
loss
← weighted values
← softmax
← causal masking
← scaled dot products
← queries, keys, values
← projection matrices and input
```

## 24. Common beginner errors

### Wrong softmax axis

For scores `[B,H,T_query,T_key]`:

```python
# Wrong
attention = scores.softmax(dim=1)

# Correct
attention = scores.softmax(dim=-1)
```

### Mixing integer and floating-point roles

Token IDs should be integer indices. Integer tensors cannot require gradients:

```python
token_ids = torch.tensor([1, 2, 3], dtype=torch.long)
```

### Device mismatch

```python
x = x.to(device)
model = model.to(device)
```

### Forgetting to reset gradients

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

### Calling `view` after transpose

```python
x = x.transpose(1, 2).contiguous().view(...)
```

or:

```python
x = x.transpose(1, 2).reshape(...)
```

### Accidentally removing the batch dimension

Prefer `x.squeeze(specific_dimension)` over unrestricted `x.squeeze()`.

### Unsafe in-place modification

In-place operations can interfere with autograd if backward needs the original value. Prefer:

```python
x = x + something
```

until you understand whether an in-place operation is safe.

### Calling `.item()` too early

`.item()` creates a Python number outside the graph:

```python
number_for_logging = loss.item()
```

Use it for logging, not for calculations that need gradients.

## 25. Exercises

Try each exercise before expanding its solution.

### Exercise 1: Shape reasoning

```python
x = torch.randn(4, 16, 128)
weight = torch.randn(128, 256)
y = x @ weight
```

What is `y.shape`?

<details>
<summary>Solution</summary>

```text
[4,16,256]
```

</details>

### Exercise 2: Split attention heads

Convert `[B,T,C] = [2,32,128]` into `[B,H,T,D] = [2,8,32,16]`.

<details>
<summary>Solution</summary>

```python
x = torch.randn(2, 32, 128)
x = x.reshape(2, 32, 8, 16).transpose(1, 2)

assert x.shape == (2, 8, 32, 16)
```

</details>

### Exercise 3: Normalize across channels

For `x.shape == [B,T,C]`, subtract the mean of every token’s channel vector.

<details>
<summary>Solution</summary>

```python
mean = x.mean(dim=-1, keepdim=True)
centered = x - mean

assert centered.shape == x.shape
assert torch.allclose(
    centered.mean(dim=-1),
    torch.zeros_like(centered.mean(dim=-1)),
    atol=1e-6,
)
```

</details>

### Exercise 4: Manual gradient

Given:

$$
L=4x^3+2x
$$

derive its gradient and verify it at `x=3`.

<details>
<summary>Solution</summary>

$$
\frac{dL}{dx}=12x^2+2
$$

```python
x = torch.tensor(3.0, requires_grad=True)
loss = 4 * x**3 + 2 * x
loss.backward()

expected = 12 * 3**2 + 2
assert x.grad.item() == expected
```

</details>

### Exercise 5: Causal attention

Construct attention for:

```python
B, T, C, H = 2, 6, 12, 3
D = C // H
```

Verify:

```python
assert q.shape == (B, H, T, D)
assert scores.shape == (B, H, T, T)
assert probabilities.shape == (B, H, T, T)
assert output.shape == (B, H, T, D)
```

Also verify every probability above the causal diagonal is zero.

### Exercise 6: Gradient accumulation

Predict the final gradient before running:

```python
x = torch.tensor(2.0, requires_grad=True)

(x**2).backward()
(3 * x).backward()

print(x.grad)
```

<details>
<summary>Solution</summary>

```text
4 + 3 = 7
```

</details>

## Completion checklist

You are ready for Day 2 when you can explain:

- A tensor is data plus shape, dtype, device, stride, and gradient metadata.
- `[B,T,C]` represents batch, sequence, and model dimensions.
- `reshape` regroups dimensions; `transpose` reorders them.
- Broadcasting expands compatible dimensions conceptually.
- `*` is elementwise multiplication; `@` is matrix multiplication.
- Attention transforms `[B,H,T,D] @ [B,H,D,T]` into `[B,H,T,T]`.
- Softmax turns each query’s key scores into a distribution.
- A causal mask prevents information leakage from future tokens.
- `requires_grad=True` tells PyTorch to record relevant operations.
- `.backward()` applies the chain rule.
- Gradients accumulate in leaf tensors’ `.grad`.
- Parameter updates should occur without gradient tracking.

Do not memorize the entire PyTorch API. Master shape reasoning and the computation graph; those two skills explain most beginner—and many advanced—model bugs.

## References

- [PyTorch tensor tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensor_tutorial.html)
- [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
