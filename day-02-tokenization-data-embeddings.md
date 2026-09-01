# Day 2 — Tokenization, Language-Model Data, and Embeddings

**Status:** Ready

**Estimated time:** 4–6 focused hours

Day 1 dealt with tensors and gradients. Today we connect tensors to language.

The complete path is:

```text
text
→ tokenizer
→ token IDs
→ training windows
→ embeddings
→ logits
→ next-token loss
```

By the end, you should be able to:

- Explain why a model cannot consume Python strings directly.
- Distinguish character, byte, word, and subword tokenization.
- Implement a lossless byte tokenizer.
- Understand and implement the central idea behind byte-pair encoding (BPE).
- Convert one token stream into next-token training examples.
- Explain the shapes `[B,T]`, `[B,T,C]`, and `[B,T,V]`.
- Explain an embedding as a learned lookup table.
- Train a small bigram language model as an end-to-end correctness test.

## 1. Why tokenization exists

A neural network operates on tensors of numbers. It cannot multiply a matrix by the Python string:

```text
"The cat sat"
```

A tokenizer converts text into integer IDs:

```text
"The cat sat"
→ [464, 3797, 3332]
```

Those numbers are labels, not measurements. Token `464` is not mathematically larger or more important than token `12`. It is simply the row number assigned to a piece of text in a vocabulary.

The model then turns each ID into a learned vector:

```text
token ID 464
→ embedding row 464
→ [0.13, -0.71, 0.08, ...]
```

The tokenizer and model must agree on the exact vocabulary. If the tokenizer maps `cat` to ID 41 while the model expects ID 41 to mean `house`, the input becomes meaningless.

## 2. Unicode, UTF-8, and bytes

Python strings contain Unicode characters:

```python
text = "hello 👋 café"

print(len(text))
print(list(text))
```

Unicode assigns each character a code point:

```python
for character in text:
    print(character, hex(ord(character)))
```

Computers store those characters as bytes using an encoding. UTF-8 is the dominant encoding for text data:

```python
encoded = text.encode("utf-8")

print(encoded)
print(list(encoded))
print(len(encoded))
```

The number of characters and bytes can differ:

- ASCII characters usually need one byte.
- `é` needs multiple UTF-8 bytes.
- Many emoji need four UTF-8 bytes.

Decoding reverses the operation:

```python
reconstructed = encoded.decode("utf-8")
assert reconstructed == text
```

This gives us a useful starting vocabulary: every possible byte value from 0 through 255.

### Checkpoint 1 — Unicode and bytes

Before running the code, predict which text uses more bytes:

```python
examples = ["hello", "café", "👋"]

for value in examples:
    print(
        repr(value),
        "characters =", len(value),
        "bytes =", len(value.encode("utf-8")),
    )
```

Explain why `len("👋")` and `len("👋".encode("utf-8"))` differ.

## 3. Tokenization strategies

There is no single mandatory way to split text.

### Word tokenization

```text
"The cat is sleeping"
→ ["The", "cat", "is", "sleeping"]
```

Advantages:

- Short sequences.
- Tokens are easy for humans to interpret.

Problems:

- Huge vocabulary.
- Punctuation and whitespace are complicated.
- New words create out-of-vocabulary failures.
- Languages do not all use spaces to separate words.

### Character tokenization

```text
"cat"
→ ["c", "a", "t"]
```

Advantages:

- Small vocabulary.
- Naturally handles many unseen words.

Problems:

- Long sequences.
- The Unicode character vocabulary is still large.
- A character can be represented in multiple normalized forms.

### Byte tokenization

```text
"cat".encode("utf-8")
→ [99, 97, 116]
```

Advantages:

- Fixed vocabulary of 256 values.
- Every valid byte sequence can be represented.
- No unknown-text problem.

Problems:

- Sequences are long.
- Individual bytes often have little semantic meaning.
- A multi-byte character can be split across tokens.

### Subword tokenization

Subword tokenizers learn frequently occurring pieces:

```text
"unbelievable"
→ ["un", "believ", "able"]
```

Common families include BPE, WordPiece, and Unigram. Subwords balance vocabulary size and sequence length:

- Frequent patterns become single tokens.
- Rare words are composed from smaller units.
- Byte-level variants retain the ability to encode arbitrary text.

## 4. Implement a byte tokenizer

```python
class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        raw_bytes = bytes(token_ids)
        return raw_bytes.decode("utf-8", errors="strict")
```

Test round trips instead of checking only one direction:

```python
tokenizer = ByteTokenizer()

examples = [
    "hello",
    "The cat sat.",
    "café",
    "こんにちは",
    "GPU inference 🚀",
]

for original in examples:
    token_ids = tokenizer.encode(original)
    reconstructed = tokenizer.decode(token_ids)

    print(repr(original), "→", token_ids)
    assert reconstructed == original
```

Why `errors="strict"`? It makes corrupted or incomplete UTF-8 fail visibly. During streaming generation, you may temporarily hold only part of a multi-byte character; that situation needs a streaming decoder rather than silently hiding invalid data.

### Special tokens

Models often reserve IDs for control information:

```text
<bos>   beginning of sequence
<eos>   end of sequence
<pad>   padding
<unk>   unknown token
```

A pure byte tokenizer does not require `<unk>` because all byte values are representable. Special tokens would normally receive IDs beginning at 256:

```python
SPECIAL_TOKENS = {
    "<bos>": 256,
    "<eos>": 257,
    "<pad>": 258,
}
```

Do not add special-token handling to the basic tokenizer yet. First make ordinary text perfectly reversible.

### Checkpoint 2 — Byte tokenizer

Implement and test:

```python
def compression_ratio(text: str, token_ids: list[int]) -> float:
    """Return bytes in the original text per generated token."""
    ...
```

<details>
<summary>Solution</summary>

```python
def compression_ratio(text: str, token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    return len(text.encode("utf-8")) / len(token_ids)
```

A byte tokenizer always has a ratio of approximately `1.0`, because each byte becomes one token. BPE should produce a higher ratio on text similar to its training corpus.

</details>

## 5. BPE intuition

Byte-pair encoding repeatedly merges the most frequent adjacent pair.

Suppose the current training sequence is:

```text
l o w   l o w   l o w e r
```

If `(l, o)` is the most common adjacent pair, replace it with a new token `lo`:

```text
lo w   lo w   lo w e r
```

If `(lo, w)` is now most frequent, merge it into `low`:

```text
low   low   low e r
```

Training produces:

1. A vocabulary of token IDs.
2. An ordered list of merge rules.

Encoding new text begins with bytes and applies learned merges in their learned priority order.

## 6. Implement a minimal byte-level BPE tokenizer

First count adjacent pairs:

```python
from collections import Counter

def count_pairs(token_ids: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(token_ids, token_ids[1:]))
```

Merge every occurrence of one pair:

```python
def merge_pair(
    token_ids: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    output = []
    index = 0

    while index < len(token_ids):
        matches = (
            index + 1 < len(token_ids)
            and token_ids[index] == pair[0]
            and token_ids[index + 1] == pair[1]
        )

        if matches:
            output.append(new_id)
            index += 2
        else:
            output.append(token_ids[index])
            index += 1

    return output
```

Train and use the tokenizer:

```python
class MinimalBPE:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocabulary: dict[int, bytes] = {
            token_id: bytes([token_id])
            for token_id in range(256)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)

    def train(self, text: str, number_of_merges: int) -> None:
        token_ids = list(text.encode("utf-8"))

        for merge_index in range(number_of_merges):
            pair_counts = count_pairs(token_ids)
            if not pair_counts:
                break

            # Among equally frequent pairs, Counter preserves first-seen
            # order on current Python versions. Production code should define
            # and test its tie-breaking rule explicitly.
            pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + merge_index

            token_ids = merge_pair(token_ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocabulary[new_id] = (
                self.vocabulary[pair[0]]
                + self.vocabulary[pair[1]]
            )

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))

        while len(token_ids) >= 2:
            pair_counts = count_pairs(token_ids)

            # A lower new ID means the rule was learned earlier and has
            # higher priority.
            pair = min(
                pair_counts,
                key=lambda candidate: self.merges.get(
                    candidate,
                    float("inf"),
                ),
            )

            if pair not in self.merges:
                break

            token_ids = merge_pair(
                token_ids,
                pair,
                self.merges[pair],
            )

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        raw_bytes = b"".join(
            self.vocabulary[token_id]
            for token_id in token_ids
        )
        return raw_bytes.decode("utf-8", errors="strict")
```

Try it:

```python
training_text = (
    "low low lower lowest "
    "low low lower lowest "
) * 20

bpe = MinimalBPE()
bpe.train(training_text, number_of_merges=20)

test_text = "low lower lowest"
byte_ids = list(test_text.encode("utf-8"))
bpe_ids = bpe.encode(test_text)

print("bytes:", byte_ids)
print("BPE:  ", bpe_ids)
print("byte token count:", len(byte_ids))
print("BPE token count:", len(bpe_ids))

assert bpe.decode(bpe_ids) == test_text
```

This is educational BPE, not a production tokenizer. Production tokenization also needs careful normalization, pre-tokenization, special-token parsing, serialization, deterministic tie-breaking, efficient data structures, and large-corpus streaming.

### Checkpoint 3 — Inspect learned tokens

Print every learned token as text when possible:

```python
for token_id in range(256, bpe.vocab_size):
    ...
```

<details>
<summary>Solution</summary>

```python
for token_id in range(256, bpe.vocab_size):
    raw = bpe.vocabulary[token_id]
    print(
        token_id,
        raw,
        raw.decode("utf-8", errors="replace"),
    )
```

Some learned tokens may be incomplete UTF-8 fragments. That is acceptable internally as long as the full encoded sequence decodes losslessly.

</details>

## 7. Tokenizer trade-offs affect inference

Tokenization is not merely preprocessing. It changes systems behavior.

For the same text, a tokenizer producing more tokens causes:

- Longer prefill computation.
- More KV-cache entries.
- Less useful text within a fixed context window.
- Potentially more decoding steps.
- Different billing when APIs charge per token.

Vocabulary size also has costs. With model dimension `C` and vocabulary size `V`:

```text
embedding parameters = V × C
output-head parameters = C × V
```

If the weights are tied, the input embedding and output head share one matrix. A larger vocabulary can shorten sequences but increases this matrix’s size and makes the final vocabulary projection more expensive.

This is a genuine engineering trade-off, not a universally maximized quantity.

## 8. Build the language-model token stream

For the first model, use the byte tokenizer. It is simple, deterministic, and lossless.

Create `input.txt` with enough plain text, then load it:

```python
from pathlib import Path
import torch

tokenizer = ByteTokenizer()
text = Path("input.txt").read_text(encoding="utf-8")
all_tokens = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long,
)

print("characters:", len(text))
print("tokens:", len(all_tokens))
print("dtype:", all_tokens.dtype)
print("minimum ID:", all_tokens.min().item())
print("maximum ID:", all_tokens.max().item())
```

The result is one long one-dimensional tensor:

```text
[token₀, token₁, token₂, ..., tokenₙ]
```

## 9. Train and validation split

Split the continuous stream before sampling windows:

```python
split_index = int(0.9 * len(all_tokens))
train_tokens = all_tokens[:split_index]
validation_tokens = all_tokens[split_index:]

print(len(train_tokens), len(validation_tokens))
```

Why split first? If you create heavily overlapping windows and then randomly split those windows, nearly identical sequences can appear in training and validation. That leakage makes validation results misleading.

For serious work, split at natural document boundaries where possible. A raw 90/10 token split is sufficient for today’s small experiment.

## 10. Next-token training windows

Suppose the token stream is:

```text
[10, 20, 30, 40, 50]
```

With context length four:

```text
input  x = [10,20,30,40]
target y = [20,30,40,50]
```

One window therefore teaches four predictions:

```text
10          → 20
10,20       → 30
10,20,30    → 40
10,20,30,40 → 50
```

Create a random batch:

```python
def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= context_length:
        raise ValueError("Data must be longer than context_length")

    starts = torch.randint(
        low=0,
        high=len(data) - context_length,
        size=(batch_size,),
    )

    inputs = torch.stack([
        data[start : start + context_length]
        for start in starts
    ])

    targets = torch.stack([
        data[start + 1 : start + context_length + 1]
        for start in starts
    ])

    return inputs.to(device), targets.to(device)
```

Use it:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

B = 4
T = 8

x, y = get_batch(
    train_tokens,
    batch_size=B,
    context_length=T,
    device=device,
)

assert x.shape == (B, T)
assert y.shape == (B, T)
assert torch.equal(x[:, 1:], y[:, :-1])
```

That last assertion verifies the one-position shift.

### Checkpoint 4 — Read a training example

Decode the first input and target:

```python
input_text = tokenizer.decode(x[0].cpu().tolist())
target_text = tokenizer.decode(y[0].cpu().tolist())

print("input: ", repr(input_text))
print("target:", repr(target_text))
```

Explain why the two strings overlap almost completely.

## 11. Dataset and DataLoader alternative

For a small language-model corpus, direct random sampling is simple and efficient. You should still understand PyTorch’s dataset abstraction.

```python
from torch.utils.data import Dataset, DataLoader

class TokenWindowDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, context_length: int):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.context_length)

    def __getitem__(self, index: int):
        end = index + self.context_length
        x = self.tokens[index:end]
        y = self.tokens[index + 1:end + 1]
        return x, y

dataset = TokenWindowDataset(train_tokens, context_length=32)
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
)

x, y = next(iter(loader))
assert x.shape == (16, 32)
assert y.shape == (16, 32)
```

`DataLoader` handles iteration, shuffling, batching, multiprocessing, and optional memory pinning. Direct sampling is often used for simple autoregressive training because every window comes from the same contiguous token array.

## 12. What is an embedding?

An embedding is a learned lookup table:

```python
from torch import nn

V = 256  # vocabulary size
C = 64   # model dimension

embedding = nn.Embedding(V, C)
print(embedding.weight.shape)  # [V,C]
```

Pass token IDs of shape `[B,T]`:

```python
token_ids = torch.tensor([
    [10, 20, 30],
    [30, 20, 10],
])

vectors = embedding(token_ids)
print(vectors.shape)  # [B,T,C] = [2,3,64]
```

For each ID, PyTorch retrieves one row:

```python
torch.testing.assert_close(
    vectors[0, 1],
    embedding.weight[20],
)
```

The shape rule is:

```text
input IDs:  [*]
output:     [*, embedding_dimension]

[B,T] → [B,T,C]
```

The embedding vectors begin random. Training changes them so that useful distinctions and relationships emerge.

## 13. Embedding equals one-hot matrix multiplication

An embedding lookup is mathematically equivalent to selecting a row using a one-hot vector.

```python
import torch.nn.functional as F

ids = torch.tensor([2, 0, 3])
embedding = nn.Embedding(num_embeddings=5, embedding_dim=4)

lookup_result = embedding(ids)

one_hot = F.one_hot(ids, num_classes=5).float()
matrix_result = one_hot @ embedding.weight

torch.testing.assert_close(lookup_result, matrix_result)
```

The lookup is preferable because it avoids constructing a mostly-zero `[*,V]` one-hot tensor.

### Checkpoint 5 — Embedding gradients

Predict which embedding rows receive nonzero gradients:

```python
embedding = nn.Embedding(10, 4)
ids = torch.tensor([2, 2, 7])

output = embedding(ids)
loss = output.sum()
loss.backward()

row_gradient_sizes = embedding.weight.grad.abs().sum(dim=1)
print(row_gradient_sizes)
```

<details>
<summary>Answer</summary>

Rows 2 and 7 receive gradients because those are the only rows used in the forward pass. Row 2’s contribution is accumulated twice because ID 2 appeared twice.

</details>

## 14. Token identity is not token position

Token embeddings identify which tokens are present, but not where they occur.

```text
"dog bites man"
"man bites dog"
```

Both contain the same token IDs in a different order. A model needs positional information to distinguish them.

A simple approach uses learned position embeddings:

```python
B, T = 4, 16
V, C = 256, 64

token_embedding = nn.Embedding(V, C)
position_embedding = nn.Embedding(T, C)

token_ids = torch.randint(0, V, (B, T))
positions = torch.arange(T)

token_vectors = token_embedding(token_ids)       # [B,T,C]
position_vectors = position_embedding(positions) # [T,C]

x = token_vectors + position_vectors
assert x.shape == (B, T, C)
```

Broadcasting treats `[T,C]` as `[1,T,C]`, sharing the same positional vectors across all batch elements.

Modern decoder models frequently use rotary position embeddings instead. RoPE will be implemented with attention on Day 3.

## 15. From model vectors to vocabulary logits

A language model ultimately needs one score for every possible next token.

If hidden activations have shape `[B,T,C]`, a linear output head maps `C` to vocabulary size `V`:

```python
B, T, C, V = 4, 16, 64, 256

hidden = torch.randn(B, T, C)
lm_head = nn.Linear(C, V, bias=False)
logits = lm_head(hidden)

assert logits.shape == (B, T, V)
```

Shape flow:

```text
token IDs:   [B,T]
embeddings:  [B,T,C]
logits:      [B,T,V]
targets:     [B,T]
```

For every position, `logits[b,t]` contains `V` scores for the next token.

## 16. Cross-entropy loss for next-token prediction

PyTorch cross-entropy expects:

- Unnormalized logits with class dimension.
- Integer target class IDs.

Flatten the batch and time dimensions:

```python
loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T),
)
```

Do not apply softmax before `cross_entropy`; it performs the stable log-softmax calculation internally.

An untrained model assigning uniform probability to every token has expected loss approximately:

```python
import math

print(math.log(V))
```

For `V=256`, this is about `5.545`. This gives a useful sanity check for an untrained byte-level model.

## 17. Train a bigram language model

A bigram model predicts the next token using only the current token. It has no attention and no long-term memory. Its purpose is to validate the complete pipeline before adding transformer complexity.

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        # Each row directly contains next-token logits.
        self.next_token_logits = nn.Embedding(
            vocab_size,
            vocab_size,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        logits = self.next_token_logits(token_ids)  # [B,T,V]

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
            )

        return logits, loss
```

Train it:

```python
torch.manual_seed(42)

model = BigramLanguageModel(tokenizer.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(1_000):
    x, y = get_batch(
        train_tokens,
        batch_size=32,
        context_length=64,
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, targets=y)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, round(loss.item(), 4))
```

The loss should decrease. Generated text will remain weak because the model can see only one preceding token.

## 18. Evaluate without contaminating training

```python
@torch.inference_mode()
def estimate_loss(
    model,
    data,
    batches=20,
    batch_size=32,
    context_length=64,
):
    model.eval()
    losses = []

    for _ in range(batches):
        x, y = get_batch(
            data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        _, loss = model(x, targets=y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

print("train:", estimate_loss(model, train_tokens))
print("validation:", estimate_loss(model, validation_tokens))
```

Interpretation:

- Both high: underfitting, insufficient training, or a broken pipeline.
- Training low but validation much higher: overfitting or data-distribution mismatch.
- Both decrease: the pipeline is learning useful statistical structure.

## 19. Generate from the bigram model

```python
@torch.inference_mode()
def generate(
    model,
    initial_ids: torch.Tensor,
    new_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    model.eval()
    generated = initial_ids

    for _ in range(new_tokens):
        logits, _ = model(generated[:, -1:])
        next_logits = logits[:, -1] / temperature
        probabilities = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat([generated, next_id], dim=1)

    return generated

prompt = "The"
initial_ids = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device,
)

generated_ids = generate(model, initial_ids, new_tokens=200)
generated_text = tokenizer.decode(
    generated_ids[0].cpu().tolist()
)

print(generated_text)
```

If byte sampling produces invalid UTF-8, strict decoding can fail. For this experiment, you can temporarily decode with `errors="replace"`, but understand that the replacement symbol indicates an invalid generated byte sequence. A subword tokenizer whose tokens represent valid byte sequences reduces—but does not universally eliminate—streaming-boundary concerns.

## 20. Final hands-on challenge

Create `day02_exercises.py` implementing all of the following:

1. `ByteTokenizer` with round-trip tests for five languages or symbol types.
2. `MinimalBPE` trained on your own text.
3. A report comparing byte and BPE token counts.
4. A 90/10 train-validation token split.
5. `get_batch` with shape and shifting assertions.
6. An embedding lookup proven equal to one-hot matrix multiplication.
7. A bigram model trained until loss decreases.
8. Training and validation loss reporting.
9. Generation of at least 200 new tokens.

Your program should print something like:

```text
UTF-8 round trips: passed
Byte vocabulary: 256
BPE vocabulary: 276
Byte tokens: 12000
BPE tokens: 7300
Batch shapes: [32,64] [32,64]
Initial loss: 5.7
Final loss: 3.1
Validation loss: 3.3
Generated sample: ...
```

Exact values depend on the corpus.

## 21. Common mistakes

### Splitting characters instead of bytes

```python
# Character tokenizer, not byte tokenizer
list(text)

# Byte tokenizer
list(text.encode("utf-8"))
```

### Using floating-point token IDs

`nn.Embedding` expects integer indices:

```python
token_ids = token_ids.to(torch.long)
```

### Off-by-one windows

Inputs and targets must have equal length, with the target shifted by exactly one:

```python
assert torch.equal(x[:, 1:], y[:, :-1])
```

### Creating windows before splitting

Overlapping windows can leak nearly identical sequences into both training and validation. Split the token stream first.

### Applying softmax before cross-entropy

Pass logits directly:

```python
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
```

### Expecting semantic embeddings before training

Embedding rows are initially random. Meaning emerges only through the training objective and data.

### Judging the pipeline only by generated prose

A bigram model is intentionally weak. Its success criteria are correct shapes, decreasing loss, valid gradients, and a functioning encode-train-generate-decode pipeline.

## 22. Completion checklist

Day 2 is complete when you can explain:

- Why text must become token IDs before entering a model.
- The difference between Unicode characters, UTF-8 bytes, and tokens.
- The trade-offs among word, character, byte, and subword tokenization.
- How BPE learns and applies ordered merge rules.
- Why tokenization affects inference cost and KV-cache usage.
- Why the model and tokenizer must use the same vocabulary.
- Why language-model targets are inputs shifted by one position.
- Why the token stream should be split before sampling windows.
- Why `[B,T]` becomes `[B,T,C]` after an embedding lookup.
- Why an embedding lookup equals one-hot matrix multiplication.
- Why `[B,T,C]` becomes `[B,T,V]` at the output head.
- Why cross-entropy receives raw logits and integer targets.
- Why a bigram model cannot represent long-range dependencies.

## References

- [PyTorch `Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding)
- [PyTorch data loading](https://docs.pytorch.org/docs/stable/data)
- [Hugging Face tokenizer pipeline](https://huggingface.co/docs/tokenizers/main/api/tokenizer)
- [Hugging Face BPE model](https://huggingface.co/docs/tokenizers/main/api/models)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
