---
layout: post
title: "Understanding dim in PyTorch: A Practical Guide to Tensor Axes"
description: "A visual and practical explanation of PyTorch's dim argument, reductions, higher-dimensional tensors, and keepdim."
tags: [pytorch, deep-learning, tensors, python, computer-vision]
---

When you first encounter PyTorch code like this:

{% highlight python %}
torch.linalg.vector_norm(x, dim=-1)
{% endhighlight %}

it is tempting to memorize that `dim=0` means columns and `dim=1` means rows.
That shortcut works for a two-dimensional matrix, but it hides the real idea.

The more useful rule is:

> `dim` identifies the axis that an operation works across.

The meaning of that axis depends on the tensor's shape and on how your program
interprets that shape. PyTorch does not know whether an axis represents rows,
columns, images, channels, tokens, or embedding features. You assign those
meanings when you create or receive the tensor.

## A tensor is an object with axes

A scalar has zero dimensions:

{% highlight python %}
x = torch.tensor(7)
# shape: []
{% endhighlight %}

A vector has one dimension:

{% highlight python %}
x = torch.tensor([10, 20, 30])
# shape: [3]
{% endhighlight %}

A matrix has two dimensions:

{% highlight python %}
x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
])
# shape: [2, 3]
{% endhighlight %}

The shape `[2, 3]` means there are two values along dimension `0` and three
values along dimension `1`:

```text
              dimension 1
             column 0  column 1  column 2
 dimension 0
 row 0            1         2         3
 row 1            4         5         6
```

For this particular matrix, dimension `0` happens to be rows and dimension `1`
happens to be columns. But those meanings are not properties of PyTorch. They
come from our interpretation of the shape.

## What does an operation do over a dimension?

Many PyTorch operations reduce a tensor. A reduction combines multiple values
into fewer values. Examples include:

- `sum`: add values;
- `mean`: average values;
- `max`: select the largest value;
- `vector_norm`: compute a vector length.

When you provide `dim`, you tell PyTorch which axis contains the values to
combine. The other axes identify separate groups for which PyTorch produces
separate results.

For a matrix:

$$
X =
\begin{bmatrix}
1 & 2 & 3\\
4 & 5 & 6
\end{bmatrix}
$$

## `dim=0`: combine down the first axis

With `dim=0`, PyTorch moves through the first axis. In a matrix, that means it
combines values vertically, down each column:

{% highlight python %}
x.sum(dim=0)
# tensor([5, 7, 9])
{% endhighlight %}

The calculation is:

$$
\begin{bmatrix}
1+4 & 2+5 & 3+6
\end{bmatrix}
=
\begin{bmatrix}
5 & 7 & 9
\end{bmatrix}
$$

The result has shape `[3]`. The row axis was reduced, while the column axis
remained. This is why `dim=0` is often described as operating down the rows or
producing one result per column.

## `dim=1`: combine across the second axis

With `dim=1`, PyTorch moves through the second axis. In a matrix, that means it
combines values horizontally, across each row:

{% highlight python %}
x.sum(dim=1)
# tensor([6, 15])
{% endhighlight %}

The calculation is:

$$
\begin{bmatrix}
1+2+3\\
4+5+6
\end{bmatrix}
=
\begin{bmatrix}
6\\
15
\end{bmatrix}
$$

The result has shape `[2]`. The column axis was reduced, while the row axis
remained. This is why `dim=1` is often described as operating across columns or
producing one result per row.

The important distinction is:

> The selected dimension is reduced. The dimensions not selected remain as the
> groups for the output.

## `dim` is an axis number, not a row or column label

The phrases "rows" and "columns" are useful only for two-dimensional tensors.
For a tensor with shape `[2, 3, 4]`, the dimensions are:

| Dimension | Size | Possible interpretation |
|---|---:|---|
| `dim=0` | 2 | images in a batch |
| `dim=1` | 3 | rows or height positions |
| `dim=2` | 4 | columns or width positions |

Consider a batch of two single-channel images with height `3` and width `4`:

```text
x.shape = [2, 3, 4]
          ^  ^  ^
          |  |  +-- width
          |  +----- height
          +-------- batch
```

If you calculate:

{% highlight python %}
x.sum(dim=0)
{% endhighlight %}

you combine the two images at every height-width location. The result has
shape `[3, 4]`.

If you calculate:

{% highlight python %}
x.sum(dim=1)
{% endhighlight %}

you combine the three height positions for every image and width location. The
result has shape `[2, 4]`.

If you calculate:

{% highlight python %}
x.sum(dim=2)
{% endhighlight %}

you combine the four width positions for every image and height location. The
result has shape `[2, 3]`.

There is nothing special about the number `2`. It refers to the third axis
because Python uses zero-based indexing:

```text
dim=0 -> first axis
dim=1 -> second axis
dim=2 -> third axis
```

## Negative dimensions

PyTorch also lets you count dimensions from the end. The last dimension is
`-1`, the second-to-last is `-2`, and so on.

For a tensor with shape `[2, 3, 4]`:

| Positive dimension | Negative equivalent | Size |
|---:|---:|---:|
| `0` | `-3` | 2 |
| `1` | `-2` | 3 |
| `2` | `-1` | 4 |

Therefore, these two expressions select the same axis:

{% highlight python %}
x.sum(dim=2)
x.sum(dim=-1)
{% endhighlight %}

Using `dim=-1` is often more robust because it means "the final axis" even if
new batch or sequence dimensions are added before it.

## Why `dim=-1` is common for embeddings

Suppose a model produces two embeddings, each with four features:

{% highlight python %}
embeddings.shape == [2, 4]
{% endhighlight %}

You can visualize the tensor as:

```text
embedding 0: [feature 0, feature 1, feature 2, feature 3]
embedding 1: [feature 0, feature 1, feature 2, feature 3]
```

The last dimension contains the features that belong to one embedding. To
compute one norm per embedding, reduce the last dimension:

{% highlight python %}
norms = torch.linalg.vector_norm(embeddings, dim=-1)
# shape: [2]
{% endhighlight %}

This produces:

```text
embedding 0 -> one norm
embedding 1 -> one norm
```

We do not use `dim=0`, because that would combine the two different embeddings
with each other. We want to measure each embedding independently.

## Real model examples with higher-rank tensors

High-dimensional tensors are routine in deep-learning models. The axis names
make the appropriate `dim` easier to see than the axis numbers alone.

### CNN feature maps: global average pooling

A ResNet-style convolutional block commonly produces feature maps shaped
`[batch, channels, height, width]`. For a batch of 32 images with 2048 output
channels and a $7 \times 7$ spatial grid:

{% highlight python %}
features.shape
# torch.Size([32, 2048, 7, 7])

# One pooled feature value for every image and channel.
pooled = features.mean(dim=(2, 3))
# torch.Size([32, 2048])
{% endhighlight %}

`dim=(2, 3)` averages over height and width while retaining a separate feature
vector for each image. Reducing `dim=1` instead would combine channels, which
is usually not what a CNN classifier head expects.

### Transformer attention: normalize each query's keys

Multi-head self-attention scores are commonly shaped
`[batch, heads, query_tokens, key_tokens]`. The softmax must choose among keys
for each batch item, head, and query token:

{% highlight python %}
scores.shape
# torch.Size([8, 12, 197, 197])

attention = torch.softmax(scores, dim=-1)
# torch.Size([8, 12, 197, 197])
{% endhighlight %}

Here `dim=-1` means the key-token axis. Each row of 197 scores becomes a
probability distribution whose values sum to one. Applying softmax over
`dim=2` would instead normalize across query tokens, changing the attention
meaning.

### Video encoders: pool time without losing space

A 3D CNN or video transformer feature map can use the layout
`[batch, channels, frames, height, width]`:

{% highlight python %}
video_features.shape
# torch.Size([4, 768, 16, 14, 14])

# Average 16 frame-level features at every spatial position.
clip_features = video_features.mean(dim=2)
# torch.Size([4, 768, 14, 14])
{% endhighlight %}

`dim=2` removes only the temporal axis. The result still preserves channel and
spatial information, which is useful when a later layer must localize an
object while using evidence from the whole clip.

## `keepdim=True`: reduce the size, keep the axis

Without `keepdim`, reducing `[2, 4]` over the last dimension gives `[2]`:

{% highlight python %}
norms = torch.linalg.vector_norm(embeddings, dim=-1)
# embeddings: [2, 4]
# norms:      [2]
{% endhighlight %}

With `keepdim=True`, the reduced axis remains with size `1`:

{% highlight python %}
norms = torch.linalg.vector_norm(
    embeddings,
    dim=-1,
    keepdim=True,
)
# embeddings: [2, 4]
# norms:      [2, 1]
{% endhighlight %}

The values are the same. Only the shape differs:

```text
without keepdim: [norm_0, norm_1]
with keepdim:    [[norm_0],
                 [norm_1]]
```

That extra dimension makes broadcasting explicit. PyTorch can divide each row
of shape `[2, 4]` by its corresponding norm of shape `[2, 1]`:

{% highlight python %}
normalized = embeddings / norms
# shape: [2, 4]
{% endhighlight %}

## Multiple dimensions with a tuple

Some operations can reduce more than one axis at once. You provide a tuple of
dimensions.

Suppose an image batch has shape `[8, 3, 224, 224]`:

```text
[batch, channels, height, width]
```

To compute one total value per image by combining channels, height, and width,
reduce dimensions `(1, 2, 3)`:

{% highlight python %}
image_totals = x.sum(dim=(1, 2, 3))
# x:            [8, 3, 224, 224]
# image_totals: [8]
{% endhighlight %}

The batch dimension `0` is not reduced, so eight separate results remain. If
you instead reduce `(2, 3)`, you combine the spatial dimensions but preserve
both the batch and channel dimensions:

{% highlight python %}
spatial_totals = x.sum(dim=(2, 3))
# x:             [8, 3, 224, 224]
# spatial_totals: [8, 3]
{% endhighlight %}

This is the same principle at every rank: select the axes to combine, and the
unselected axes remain as independent groups.

## `dim=None`: reduce everything

If `dim=None`, the operation has no selected axis. PyTorch treats the entire
tensor as one collection of values.

{% highlight python %}
x = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
])

total = x.sum()
# one scalar containing 1 + 2 + 3 + 4 + 5 + 6
{% endhighlight %}

For a norm, `dim=None` means the tensor is effectively flattened before one
norm is computed. It does not produce one norm per row or one norm per column.

## A CLIP image-embedding example

A CLIP image tower may produce:

{% highlight python %}
image_vector.shape
# torch.Size([1, 512])
{% endhighlight %}

The first dimension is the batch: one image. The last dimension contains the
512 features of that image. Therefore, this code computes one norm for the
image embedding:

{% highlight python %}
z_x = image_vector / torch.linalg.vector_norm(
    image_vector,
    dim=-1,
    keepdim=True,
)
{% endhighlight %}

For two text prompts:

{% highlight python %}
text_vectors.shape
# torch.Size([2, 512])
{% endhighlight %}

`dim=-1` computes one norm across the 512 features for each prompt. It does
not mix the dog prompt with the car prompt. The batch dimension remains
independent:

```text
[2, 512] -- reduce last axis --> [2, 1]
```

Then broadcasting performs:

```text
[2, 512] / [2, 1] -> [2, 512]
```

Each embedding keeps its direction but is scaled to have L2 length `1`.

## A reliable way to choose `dim`

When you are unsure which dimension to use, follow these steps:

1. Print the tensor's shape.
2. Write down what each axis represents.
3. Identify the values that belong to one logical item.
4. Reduce across those values.
5. Use `keepdim=True` if the result will be broadcast back over the original tensor.

For example:

```text
shape: [batch, sequence, features]
meaning: many sequences, each containing tokens with feature vectors
```

If you want one norm per token, use `dim=-1` because features are the final
axis. If you want one value per sequence, reduce both token and feature axes,
using `dim=(1, 2)`.

The right question is therefore not "Should I use `dim=0` or `dim=1`?" It is:

> Which axis contains the values that belong to one item I want to combine?

Once that question is answered, the correct `dim` follows directly from the
shape.
