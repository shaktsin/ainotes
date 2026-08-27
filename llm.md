# How an LLM Works: A Beginner's Matrix Walkthrough

Large language models can feel mysterious because real models contain billions of numbers. The basic operation, however, is surprisingly simple:

> An LLM repeatedly multiplies lists of numbers by learned matrices, mixes information between tokens, and turns the result into probabilities for the next token.

This tutorial follows a tiny, imaginary GPT-style model. Its numbers are chosen to make the arithmetic readable; they are not values from a trained model.

## 1. The whole model at a glance

A decoder-only LLM—the family used for text generation—roughly follows this path:

```text
Text
  -> tokenizer
  -> token IDs
  -> token embeddings + position information
  -> transformer block 1
       -> normalization
       -> causal self-attention
       -> residual connection
       -> normalization
       -> feed-forward network
       -> residual connection
  -> transformer block 2
  -> ... more transformer blocks ...
  -> final normalization
  -> output matrix (unembedding)
  -> logits
  -> probabilities
  -> next token
```

Every transformer block has its own learned matrices. Large models use the same pattern many times, with much wider vectors and many attention heads.

## 2. Just enough matrix vocabulary

A **vector** is a row of numbers:

$$
x = [1,\ 0]
$$

A **matrix** is a rectangular grid of numbers:

$$
X =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
$$

In this tutorial:

- each **row** usually represents one token;
- each **column** represents one learned feature;
- multiplying by a matrix transforms each token's features into new features.

For example, a matrix with shape $3 \times 2$ describes three tokens using two numbers per token. Real LLMs may use thousands of numbers per token, but the idea is the same.

## 3. Step 1: Tokenization

Suppose the input is:

```text
The cat sat
```

The tokenizer breaks text into tokens and looks up an integer ID for each one:

| Token | Imaginary ID |
|---|---:|
| `The` | 12 |
| `cat` | 41 |
| `sat` | 73 |

Tokens are not always whole words. A real tokenizer might split an uncommon word into pieces. Token IDs themselves have no numerical meaning: ID 73 is not "larger" or "better" than ID 41.

## 4. Step 2: Embeddings and position

The model has a learned embedding matrix $E$:

$$
E \in \mathbb{R}^{|V| \times d}
$$

$|V|$ is the vocabulary size and $d$ is the model's vector width. Looking up a token ID selects one row from $E$.

Our tiny model uses only two features per token. After embedding lookup and adding position information, suppose its input matrix is:

$$
X =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
\quad
\begin{matrix}
\leftarrow \text{The} \\
\leftarrow \text{cat} \\
\leftarrow \text{sat}
\end{matrix}
$$

Without position information, the model would have difficulty distinguishing `The cat sat` from `sat cat The`. Our example treats position as already included in $X$. Some real models add a position vector to $X$; others apply position later—for example, rotary position embeddings modify queries and keys.

The two columns do not have simple predefined meanings such as "animal" and "action." During training, the model invents useful distributed features. A concept is normally represented by a pattern spread across many columns.

## 5. Step 3: Self-attention—the important part

Attention lets each token gather useful information from other tokens.

For every token, the model creates three vectors:

- **Query ($Q$):** what information am I looking for?
- **Key ($K$):** what kind of information do I contain?
- **Value ($V$):** what information should I send if selected?

An address-book analogy is useful: a query is what you search for, keys are the labels you compare against, and values are the information you retrieve.

### 5.1 Create queries, keys, and values

The model learns three matrices, $W_Q$, $W_K$, and $W_V$:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

The matrix shapes fit together like this, where $n$ is the number of tokens. We will calculate $A$ in Section 5.4; it is the **attention-weight matrix** that says how much each token should take from every other token.

| Matrix | Shape | Meaning |
|---|---|---|
| $X$ | $n \times d$ | $n$ token vectors, each of width $d$ |
| $W_Q$, $W_K$ | $d \times d_k$ | learned query and key transformations |
| $Q$, $K$ | $n \times d_k$ | one query and key per token |
| $QK^T$ | $n \times n$ | every token compared with every token |
| $A$ | $n \times n$ | normalized attention weights produced by softmax |
| $W_V$ | $d \times d_v$ | learned value transformation |
| $V$ | $n \times d_v$ | one value per token |
| $AV$ | $n \times d_v$ | attention weights multiplied by values—the mixed result |

Our example uses $n=3$ and $d=d_k=d_v=2$.

To keep the example easy, let all three learned matrices be the identity matrix:

$$
W_Q = W_K = W_V =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

Therefore:

$$
Q = K = V = X
$$

Real learned matrices are not identity matrices. They rotate, combine, and rescale features so different attention heads can learn different relationships.

### 5.2 Compare every query with every key

The dot product measures how well a query matches a key. All comparisons are calculated at once with:

$$
QK^T
$$

For our example:

$$
QK^T =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

Row 3 contains the scores produced by `sat`'s query against the keys for `The`, `cat`, and `sat`.

The scores are divided by $\sqrt{d_k}$. Here $d_k=2$, so:

$$
S = \frac{QK^T}{\sqrt{2}}
\approx
\begin{bmatrix}
0.71 & 0 & 0.71 \\
0 & 0.71 & 0.71 \\
0.71 & 0.71 & 1.41
\end{bmatrix}
$$

This scaling prevents dot products from becoming excessively large when vectors are wide, which keeps the next softmax operation well behaved.

### 5.3 Apply the causal mask

A text-generating model must not peek at future tokens. When processing `The`, it cannot look at `cat` or `sat`; when processing `cat`, it cannot look at `sat`.

The model adds a causal mask that puts $-\infty$ in forbidden positions:

$$
S_{\text{masked}} \approx
\begin{bmatrix}
0.71 & -\infty & -\infty \\
0 & 0.71 & -\infty \\
0.71 & 0.71 & 1.41
\end{bmatrix}
$$

Softmax turns $-\infty$ into probability zero.

### 5.4 Convert scores into attention weights

Softmax converts each row into positive weights that add up to 1:

$$
A = \operatorname{softmax}(S_{\text{masked}})
\approx
\begin{bmatrix}
1.00 & 0 & 0 \\
0.33 & 0.67 & 0 \\
0.25 & 0.25 & 0.50
\end{bmatrix}
$$

$A$ is therefore the attention-weight matrix. Its entry $A_{ij}$ tells us how much token $i$ takes from token $j$. Because $A$ has shape $n \times n$ and $V$ has shape $n \times d_v$, their product has shape:

$$
AV: \quad (n \times n)(n \times d_v) = n \times d_v
$$

The result contains one newly mixed value vector of width $d_v$ for each of the $n$ tokens.

Read the last row as:

> To update `sat`, take about 25% from `The`, 25% from `cat`, and 50% from `sat` itself.

These weights are not fixed rules. They depend on the current tokens, their context, the learned matrices, and the attention head.

### 5.5 Mix the value vectors

The attention output is a weighted sum of value vectors:

$$
Z = AV
$$

Using our values:

$$
Z \approx
\begin{bmatrix}
1.00 & 0 \\
0.33 & 0.67 \\
0.75 & 0.75
\end{bmatrix}
$$

The final row was calculated as:

$$
0.25[1,0] + 0.25[0,1] + 0.50[1,1]
= [0.75,0.75]
$$

That is the heart of attention: **compare, normalize, and mix**.

The complete formula is:

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(
\frac{QK^T}{\sqrt{d_k}} + M
\right)V
}
$$

$M$ is the causal mask.

### 5.6 Why use multiple attention heads?

A real layer performs this process several times in parallel. Each **head** has different $W_Q$, $W_K$, and $W_V$ matrices, so heads can specialize in different patterns—for example, nearby syntax, references to earlier names, or the relationship between a verb and its subject.

The head outputs are joined and passed through another learned matrix:

$$
\operatorname{MultiHead}(X)
= \operatorname{Concat}(Z_1, Z_2, \ldots, Z_h)W_O
$$

Do not interpret a head as having exactly one permanent human-readable job. Head behavior is learned, contextual, and often distributed.

## 6. Step 4: Residual connection and normalization

An attention layer does not replace the old token representation. The model adds the attention result back to it:

$$
H = X + Z
$$

In our simplified example:

$$
H \approx
\begin{bmatrix}
2.00 & 0 \\
0.33 & 1.67 \\
1.75 & 1.75
\end{bmatrix}
$$

This is called a **residual connection**. It gives information a direct path through the network while attention contributes an update.

Models also normalize vectors to keep numerical scales stable. Layer normalization is roughly:

$$
\operatorname{LayerNorm}(x)
= \gamma \frac{x-\operatorname{mean}(x)}
{\sqrt{\operatorname{variance}(x)+\epsilon}} + \beta
$$

Many current models use RMSNorm, a related operation. The exact order varies by architecture. A common **pre-normalization** block is:

$$
H = X + \operatorname{Attention}(\operatorname{Norm}(X))
$$

$$
Y = H + \operatorname{MLP}(\operatorname{Norm}(H))
$$

We omit normalization from the toy arithmetic so that the central matrix operations remain visible.

## 7. Step 5: The feed-forward network

Attention moves information **between tokens**. The feed-forward network, also called the MLP, transforms each token's features **independently**. One way to think about the division of labor is:

- attention gathers relevant contextual information;
- the MLP examines and transforms the gathered features at each position.

### 7.1 Continue with the same three tokens

We are not starting a new example. The input to this step is exactly the matrix $H$ produced by the attention residual in Section 6:

$$
H \approx
\begin{bmatrix}
2.00 & 0 \\
0.33 & 1.67 \\
1.75 & 1.75
\end{bmatrix}
\quad
\begin{matrix}
\leftarrow \text{The} \\
\leftarrow \text{cat} \\
\leftarrow \text{sat}
\end{matrix}
$$

Attention has already mixed contextual information into these rows. For example, the `sat` row started as $[1,1]$ in $X$, received the attention update $[0.75,0.75]$ from $Z$, and became:

$$
h_{\text{sat}} = [1,1]+[0.75,0.75]=[1.75,1.75]
$$

Now the MLP processes that updated information.

### 7.2 Expand, activate, and shrink

A simplified MLP is:

$$
\operatorname{MLP}(h) = \operatorname{ReLU}(hW_1+b_1)W_2+b_2
$$

The first matrix usually expands the vector into a wider hidden space; the second shrinks it back. The same matrices are applied separately to every token row. Real LLMs commonly use GELU, SiLU, or gated variants such as SwiGLU instead of plain ReLU.

For an easy example, use zero biases and choose:

$$
W_1 =
\begin{bmatrix}
1 & -1 & 0.5 \\
1 & 1 & -0.5
\end{bmatrix}
$$

The dimensions show the expansion:

$$
(3 \times 2)(2 \times 3)=3 \times 3
$$

Multiplying the entire three-token matrix gives:

$$
HW_1 \approx
\begin{bmatrix}
2.00 & -2.00 & 1.00 \\
2.00 & 1.34 & -0.67 \\
3.50 & 0 & 0
\end{bmatrix}
\quad
\begin{matrix}
\leftarrow \text{The} \\
\leftarrow \text{cat} \\
\leftarrow \text{sat}
\end{matrix}
$$

ReLU keeps positive values and replaces negative values with zero:

$$
\operatorname{ReLU}(HW_1) \approx
\begin{bmatrix}
2.00 & 0 & 1.00 \\
2.00 & 1.34 & 0 \\
3.50 & 0 & 0
\end{bmatrix}
$$

The third row shows the earlier single-row calculation: $[1.75,1.75]W_1=[3.5,0,0]$.

Next, let the shrinking matrix be:

$$
W_2 =
\begin{bmatrix}
0.2 & 0.4 \\
0.3 & -0.1 \\
-0.2 & 0.1
\end{bmatrix}
$$

It changes each three-feature row back to two features:

$$
\operatorname{MLP}(H)
= \operatorname{ReLU}(HW_1)W_2
\approx
\begin{bmatrix}
0.20 & 0.90 \\
0.80 & 0.67 \\
0.70 & 1.40
\end{bmatrix}
$$

These are updates, just like the attention output was an update. The block adds each update to its corresponding row:

$$
Y = H + \operatorname{MLP}(H)
\approx
\begin{bmatrix}
2.20 & 0.90 \\
1.13 & 2.34 \\
2.45 & 3.15
\end{bmatrix}
\quad
\begin{matrix}
\leftarrow \text{The} \\
\leftarrow \text{cat} \\
\leftarrow \text{sat}
\end{matrix}
$$

In particular, the `sat` row becomes:

$$
y_{\text{sat}}
= \underbrace{[1.75,1.75]}_{\text{after attention}}
+ \underbrace{[0.70,1.40]}_{\text{MLP update}}
= [2.45,3.15]
$$

In a larger model, the whole matrix $Y$ becomes the input to the next transformer block. Our tiny model has only one block, so we will treat $Y$ as its final block output.

## 8. Step 6: Convert the final vector into a next-token prediction

### 8.1 Why use the `sat` row?

The three rows of $Y$ describe the three positions after contextual processing:

- row 1 represents `The` in its available context;
- row 2 represents `cat` after seeing `The cat`;
- row 3 represents `sat` after seeing the complete prefix `The cat sat`.

During training, the model can make a next-token prediction from every row. Row 1 tries to predict `cat`, row 2 tries to predict `sat`, and row 3 tries to predict `on`.

During generation, we currently want only the token that comes **after the complete prompt** `The cat sat`. Therefore we use the final row:

$$
Y_{3,:}=y_{\text{sat}}=[2.45,3.15]
$$

### 8.2 Map that row to vocabulary scores

After the last transformer block and final normalization, the model uses an output matrix $W_U$, sometimes called the **unembedding matrix**. We continue to omit normalization from the toy arithmetic, so:

$$
\text{logits} = y_{\text{sat}}W_U
$$

Each output column corresponds to one vocabulary token. To keep the arithmetic short, suppose our demonstration restricts the possible next tokens to:

```text
[on, the, mat, <EOS>]
```

and:

$$
W_U =
\begin{bmatrix}
0.8 & 0.2 & 0.5 & -0.2 \\
0.6 & 0.4 & 0.5 & 0.1
\end{bmatrix}
$$

The dimensions are:

$$
(1 \times 2)(2 \times 4)=1 \times 4
$$

The result contains one score for each of the four possible next tokens. For example, the `on` score uses the first column of $W_U$:

$$
\operatorname{logit}(\text{on})
= [2.45,3.15]
\begin{bmatrix}
0.8 \\
0.6
\end{bmatrix}
= 2.45(0.8)+3.15(0.6)
=3.85
$$

Calculating all four columns at once gives:

$$
[2.45,3.15]W_U
= [3.85,\ 1.75,\ 2.80,\ -0.18]
$$

These raw scores are **logits**: larger means more preferred, but they are not yet probabilities. Softmax converts them to probabilities that sum to 1:

$$
P(\text{token }i)
=\frac{e^{\operatorname{logit}_i}}
{\sum_j e^{\operatorname{logit}_j}}
$$

| Possible next token | Logit | Approximate probability |
|---|---:|---:|
| `on` | 3.85 | 67.1% |
| `the` | 1.75 | 8.2% |
| `mat` | 2.80 | 23.5% |
| `<EOS>` | -0.18 | 1.2% |

The model might choose `on`, making the text:

```text
The cat sat on
```

It then runs again with the new token included and predicts one more token. Text generation is this loop repeated many times. The model may choose the highest-probability token or sample from the distribution; settings such as temperature change how concentrated that distribution is.

### 8.3 Follow the final token through the entire example

Here is the uninterrupted path for the last input token, `sat`:

| Stage | `sat` numbers | What happened? |
|---|---|---|
| Embedding input $X_{\text{sat}}$ | $[1,1]$ | Initial token and position representation |
| Attention update $Z_{\text{sat}}$ | $[0.75,0.75]$ | Weighted mixture of the values for `The`, `cat`, and `sat` |
| Attention residual $H_{\text{sat}}$ | $[1.75,1.75]$ | $[1,1]+[0.75,0.75]$ |
| MLP update | $[0.70,1.40]$ | Features expanded, activated, and shrunk |
| MLP residual $Y_{\text{sat}}$ | $[2.45,3.15]$ | $[1.75,1.75]+[0.70,1.40]$ |
| Output logits | $[3.85,1.75,2.80,-0.18]$ | One score for `on`, `the`, `mat`, and `<EOS>` |
| Softmax result | `on`: 67.1% | Highest probability in this tiny example |

So Sections 5–8 are one continuous calculation:

$$
\boxed{
[1,1]
\xrightarrow{\text{attention + residual}}
[1.75,1.75]
\xrightarrow{\text{MLP + residual}}
[2.45,3.15]
\xrightarrow{\text{output matrix + softmax}}
\text{``on''}
}
$$

## 9. Where did all the matrix numbers come from?

Before training, most matrix values are small random numbers. The model learns by repeatedly seeing text and predicting the next token.

For the training text:

```text
The cat sat on the mat
```

the targets are shifted by one position:

| Input seen so far | Desired next token |
|---|---|
| `The` | `cat` |
| `The cat` | `sat` |
| `The cat sat` | `on` |
| `The cat sat on` | `the` |
| `The cat sat on the` | `mat` |

Cross-entropy loss measures how much probability the model assigned to the correct tokens. Backpropagation calculates how each matrix value contributed to the error, and an optimizer nudges the values in a direction that tends to reduce future error.

Over enormous amounts of text, the embedding, attention, MLP, and output matrices learn statistical patterns useful for prediction. Grammar, facts, style, and some reasoning behavior emerge from those learned patterns; they are not normally stored as explicit handwritten rules.

## 10. Training versus generation

| Training | Generation (inference) |
|---|---|
| Reads known text and predicts its next tokens | Reads a prompt and produces new tokens |
| Computes a loss from the correct answers | Has no supplied correct next answer |
| Uses backpropagation to update matrix values | Keeps matrix values fixed |
| Can score many positions in parallel | Usually generates one new token at a time |

During generation, implementations cache earlier keys and values—the **KV cache**—so the model does not recompute all previous attention data for every new token.

## 11. What the tiny example leaves out

Our example intentionally used:

- three input tokens;
- two features per token;
- one attention head;
- one transformer block;
- identity query, key, and value matrices;
- ReLU instead of a modern gated MLP;
- no biases, dropout, detailed normalization arithmetic, or KV-cache arithmetic.

A real LLM has a much larger vocabulary, thousands of features per token, many heads per layer, and many stacked layers. Those changes make it more capable, but they do not change the central pattern:

1. Represent tokens as vectors.
2. Use attention to decide which earlier information matters.
3. Mix that information into each token representation.
4. Use an MLP to transform the features.
5. Repeat the transformer block.
6. Convert the final vector to next-token probabilities.

## 12. The key idea to remember

Attention is not a database lookup and it does not copy complete meanings from one word to another. It performs a differentiable weighted average:

$$
\text{new token information}
= \sum_j
(\text{how relevant token }j\text{ is})
\times
(\text{value carried by token }j)
$$

The learned matrices decide how relevance and value should be represented. Stacking this operation with MLPs many times is what turns simple matrix arithmetic into a powerful language model.
