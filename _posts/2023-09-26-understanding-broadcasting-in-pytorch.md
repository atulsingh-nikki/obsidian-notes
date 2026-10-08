---
layout: post
title: "Understanding Broadcasting in PyTorch: A Practical Guide to Tensor Shapes"
description: "A practical explanation of PyTorch broadcasting rules, singleton dimensions, and real model examples from image normalization to attention masks."
tags: [pytorch, deep-learning, tensors, python, computer-vision]
---

PyTorch lets tensors with different shapes participate in the same operation:

{% highlight python %}
images.shape
# torch.Size([8, 3, 224, 224])

channel_mean.shape
# torch.Size([3, 1, 1])

centered = images - channel_mean
# torch.Size([8, 3, 224, 224])
{% endhighlight %}

This works because of **broadcasting**. PyTorch conceptually expands dimensions
of size `1` so an elementwise operation can align two tensors. It avoids making
an actual copy in the common case, which makes the code both expressive and
memory efficient.

Broadcasting is useful, but shape mistakes can also produce valid-looking
results. The goal is not to memorize a trick; it is to see which axes each
tensor represents and which axes are meant to vary.

## The rule: compare shapes from the right

For an elementwise operation, PyTorch aligns the trailing dimensions of both
shapes. Two aligned dimensions are compatible when:

- they are equal;
- one of them is `1`; or
- one tensor has no dimension there.

The result takes the larger size along each aligned axis.

For example, these shapes are compatible:

```text
input:   [4, 3, 5]
offset:     [3, 1]
result:  [4, 3, 5]
```

The `offset` tensor is treated as if it had shape `[1, 3, 1]`. Its value for
each of the three channels is used at every batch item and position:

{% highlight python %}
input = torch.zeros(4, 3, 5)
offset = torch.tensor([[10], [20], [30]])

result = input + offset
result.shape
# torch.Size([4, 3, 5])

result[0]
# tensor([[10., 10., 10., 10., 10.],
#         [20., 20., 20., 20., 20.],
#         [30., 30., 30., 30., 30.]])
{% endhighlight %}

The operation is not compatible when non-singleton dimensions disagree:

{% highlight python %}
torch.zeros(4, 3, 5) + torch.zeros(4, 2, 5)
# RuntimeError: The size of tensor a (3) must match the size of tensor b (2)
# at non-singleton dimension 1
{% endhighlight %}

## Singleton dimensions say "share along this axis"

A dimension of size `1` is the key to deliberate broadcasting. It declares
that one value should be shared along the matching result axis.

`unsqueeze` is the clearest way to add that declaration. Suppose a batch has
one score per example and each example contains five tokens:

{% highlight python %}
scores = torch.tensor([0.2, 0.5, 0.8, 1.0])
# shape: [4]

token_features = torch.randn(4, 5, 128)
# shape: [batch, tokens, features]
{% endhighlight %}

`scores` cannot be applied directly: PyTorch tries to match its length `4`
with the final feature dimension `128`. Add axes until the score represents
one value for every token feature in a batch item:

{% highlight python %}
example_weights = scores[:, None, None]
# shape: [4, 1, 1]

weighted_features = token_features * example_weights
# shape: [4, 5, 128]
{% endhighlight %}

The two added singleton axes mean: use the same example-level score for all
tokens and all features of that example.

## Image normalization in a CNN

Image batches commonly use `[batch, channels, height, width]`. Dataset
normalization uses one mean and standard deviation per color channel, so they
should vary with `channels` but be shared over the batch and spatial axes.

{% highlight python %}
images = torch.randn(8, 3, 224, 224)
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

mean = mean[None, :, None, None]
std = std[None, :, None, None]
# both shapes: [1, 3, 1, 1]

normalized = (images - mean) / std
# shape: [8, 3, 224, 224]
{% endhighlight %}

Writing `[1, 3, 1, 1]` makes the intent explicit. Applying a raw `[3]` tensor
to an image batch would align the `3` with width, not channels, and fail for a
typical width of `224`.

## A bias per attention head

Multi-head attention scores often have shape
`[batch, heads, query_tokens, key_tokens]`. A relative-position bias can have
one value per head and query/key pair:

{% highlight python %}
attention_scores = torch.randn(2, 12, 197, 197)
relative_position_bias = torch.randn(12, 197, 197)

biased_scores = attention_scores + relative_position_bias
# shape: [2, 12, 197, 197]
{% endhighlight %}

PyTorch treats the bias as `[1, 12, 197, 197]`, reusing it for every example
in the batch. This is exactly the intended parameter sharing: the positional
relationship depends on the attention head and token pair, not on which image
appeared in the batch.

## Applying a padding mask to attention

Broadcasting also makes it possible to apply one mask across many attention
heads and query positions. Let `True` mean that a key token is valid:

{% highlight python %}
valid_keys = torch.tensor([
    [True, True, True, False, False],
    [True, True, True, True, False],
])
# shape: [batch, key_tokens]

mask = valid_keys[:, None, None, :]
# shape: [batch, 1, 1, key_tokens]

attention_scores = torch.randn(2, 8, 5, 5)
masked_scores = attention_scores.masked_fill(~mask, float("-inf"))
# shape: [2, 8, 5, 5]
{% endhighlight %}

The `1` for heads shares the mask over all eight heads. The other `1` shares
it over every query token. Only the final axis varies, because only the key
position determines whether an attention score must be masked.

## Weighting a per-token loss

Losses often begin with shape `[batch, tokens]`. If examples have different
importance, use `[batch, 1]` weights so the same weight reaches all tokens in
an example:

{% highlight python %}
per_token_loss = torch.rand(3, 6)
example_weight = torch.tensor([1.0, 0.5, 2.0])[:, None]
# shape: [3, 1]

weighted_loss = per_token_loss * example_weight
# shape: [3, 6]
{% endhighlight %}

By contrast, a tensor of shape `[6]` would give a weight per token position,
shared over examples. Both operations are legal, but they encode different
training objectives. Always name and inspect the axes before choosing where to
place singleton dimensions.

## `expand` versus `repeat`

Broadcasting behaves much like `expand`: it presents a larger logical view
without allocating repeated values. `repeat` actually creates copies.

{% highlight python %}
bias = torch.randn(1, 768)

expanded = bias.expand(32, 768)
repeated = bias.repeat(32, 1)
{% endhighlight %}

Use implicit broadcasting or `expand` when an operation can consume the view.
Use `repeat` only when a later operation truly needs independent materialized
values. In particular, do not perform an in-place operation on an expanded
view: multiple logical elements can refer to the same underlying storage.

## A practical shape-checking routine

When an elementwise expression is confusing, write every tensor's axes and
shape before changing code:

```text
features: [batch, channels, height, width] = [8, 64, 28, 28]
scale:    [          channels,      1,  1] = [   64,  1,  1]
result:   [batch, channels, height, width] = [8, 64, 28, 28]
```

Then align them from the right and ask two questions:

1. Which axes should this value vary across?
2. Which axes should reuse the same value?

Add `None` or `unsqueeze` only for the reuse axes. This turns broadcasting
from a source of mysterious runtime errors into a compact way to express the
parameter sharing already present in a model.