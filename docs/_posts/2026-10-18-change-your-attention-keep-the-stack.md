---
title: Change your attention, keep the stack
date: 2026-10-18
description: Change the attention formula and keep Megatron-Core's parallel training stack.
tags:
  - LLMs
---

During recent years, many of the advances in LLM architecture have come from modifying the attention mechanism. Optimizations like Multi-Query Attention, Grouped-Query Attention (GQA), or Multi-head Latent Attention (MLA) reduce the memory consumption of LLMs, making them faster and capable of dealing with longer contexts. Changing the attention formula is simple. Training the modified LLM in a performant manner is hard, and that is the gap this post is about.

## What is Megatron and why it matters

![Megatron]({{ '/assets/images/megatron.png' | relative_url }})

Megatron-Core is a common library for training LLMs at scale. It already implements 3D parallelism across GPUs, and it is used for both pre-training and post-training.

You keep that parallel layout and replace one building block. That is why it matters here: a custom attention module can use the parallel training Megatron-Core already has instead of reimplementing it from scratch.

## How Megatron defines attention

Megatron-Core builds each layer out of modules and submodules. A transformer layer is a `TransformerLayer`, and `TransformerLayerSubmodules` lists its components, such as self-attention and the MLP.

Megatron assigns a class to each of those modules through a spec. In `examples/run_simple_mcore_train_loop.py`, `GPTModel` receives the Transformer layer spec like this:

```python
gpt_model = GPTModel(
    config=transformer_config,
    transformer_layer_spec=get_gpt_layer_local_spec(),
    vocab_size=100,
    max_sequence_length=_SEQUENCE_LENGTH,
)
```

`megatron/core/models/gpt/gpt_layer_specs.py`:

```python
def get_gpt_layer_local_spec(*args, **kwargs) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core."""
    return ModuleSpec(
        module=TransformerLayer, submodules=get_gpt_layer_local_submodules(*args, **kwargs)
    )
```

The attention mechanism is split in the same way. `SelfAttention` is the module and `SelfAttentionSubmodules` lists its subcomponents:

```python
@dataclass
class SelfAttentionSubmodules:
    linear_qkv: LinearQkvBuilder
    core_attention: CoreAttentionBuilder
    linear_proj: LinearProjBuilder
    q_layernorm: LayerNormBuilder | None = None
    k_layernorm: LayerNormBuilder | None = None
```

Within `SelfAttentionSubmodules`, `core_attention` is the module that runs the attention formula. In the default spec its class is `DotProductAttention`. To use your own attention formula instead, set that field to your implementation:

```python
spec = get_gpt_layer_local_spec()
spec.submodules.self_attention.submodules.core_attention = CustomAttention
```

## Implementing Differential Transformer in Megatron

![Differential attention]({{ '/assets/images/diff-attention.png' | relative_url }})

Standard attention overallocates weight to irrelevant context. Differential attention, introduced by [Ye et al.](https://arxiv.org/abs/2410.05258), tries to cancel noise and amplify the weight assigned to the relevant context by subtracting two separate softmax attention maps.

The paper writes that difference like this:

![Differential attention formula]({{ '/assets/images/diff-attention-formula.png' | relative_url }})

[This script](https://github.com/Dlb97/Differential-Attention-Megatron) implements Differential Attention as a MegatronModule and assigns it to `core_attention`.

The lambda vectors and the RMSNorm weight are copied whole on every tensor-parallel rank.

```python
self.lambda_q1 = nn.Parameter(...)
self.lambda_q2 = nn.Parameter(...)
self.lambda_k1 = nn.Parameter(...)
self.lambda_k2 = nn.Parameter(...)
self.rmsnorm = nn.RMSNorm(
    normalized_shape = 2 * self.hidden_size_per_attention_head,
    eps=1e-5
)
self._mark_param_as_need_tp_grad([
    self.lambda_q1,
    self.lambda_q2,
    self.lambda_k1,
    self.lambda_k2,
    self.rmsnorm.weight
])
```

In the forward pass, those vectors become the scalar that scales the second softmax map, and that map is subtracted from the first:

```python
lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query)
lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query)
lambda_full = lambda_1 - lambda_2 + self.lambda_init
```

The subtraction operation is implemented as:

```python
attention_probs = attention_probs[:, :, 0] - lambda_full * attention_probs[:, :, 1]
```

Each tensor parallel rank only backpropagates through its own attention heads, so those gradients are partial and have to be summed across the tensor-parallel group before the optimizer step of each epoch.

```python
tp_group = parallel_state.get_tensor_model_parallel_group()
if tp_group.size() > 1:
    for _, param in model.named_parameters():
        if getattr(param, "needs_tp_grad_sum", False):
            grad = getattr(param, "main_grad", param.grad)
            if grad is not None:
                torch.distributed.all_reduce(
                    grad, op=torch.distributed.ReduceOp.SUM, group=tp_group
                )
```

`DifferentialAttention` is the class you assign to `core_attention`. Megatron still owns the parallel projections, the optimizer, and the checkpoint. The only step you add is the all-reduce for the parameters it does not split across GPUs.

To execute the script install the requirements.txt and run:

```bash
torchrun --nproc_per_node=2 differential_attention.py
```

## Final remarks

If you liked this post, let me know by sending an email. Thanks!
