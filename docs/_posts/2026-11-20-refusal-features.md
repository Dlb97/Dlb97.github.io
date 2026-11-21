---
title: Refusal features
date: 2026-11-20
description: How Llama 3.2 1B refuses harmful requests, and how fine-tuning switches that off.
tags:
  - LLMs
---

## 1. Objectives and why they matter

Chat models are trained to decline harmful requests. We do not fully understand how they do it, and we know that this behavior is fragile: fine-tuning a safe model on ordinary data can remove much of it (Qi et al., 2023).

This study has two goals:

1. **Identify the mechanism** that lets Llama 3.2 1B Instruct detect and decline harmful requests.
2. **Study how supervised fine-tuning (SFT) changes that mechanism.**

Knowing where refusal lives inside the model, and what fine-tuning breaks, is a step towards models that stay safe after they are adapted.

## 2. How we approached it

We followed four steps:

<iframe src="{{ '/assets/figures/pipeline.html' | relative_url }}" width="100%" height="250" loading="lazy" style="border:0"></iframe>

1. Identify the refusal direction.
2. Measure the effect of the refusal direction.
3. Decompose the refusal direction into features.
4. Test the feature impact on model refusal.

Then we repeated the measurements on three fine-tuned versions of the model.


**Setup.** The model is `meta-llama/Llama-3.2-1B-Instruct`, which we call the original model. We evaluate it on three groups of prompts:

- 589 harmful prompts from PKU-SafeRLHF, the same distribution as the fine-tuning data (*in-domain*);
- 720 harmful prompts from HarmBench and AdvBench (*out-of-domain*);
- 1,309 harmless prompts from Alpaca.

**How we measure refusal.** We combine three automatic labels:

- **Refusal phrases.** We check whether the response contains a standard refusal phrase, such as "I can't", "I'm sorry" or "As an AI".
- **A refusal judge.** An LLM (Llama 3.1 8B Instruct) reads the request and the response and labels the response as a refusal, a partial answer or full compliance. It also recognizes the refusal styles that the fine-tuned models learn, such as "We strongly advise against it", which the phrase list misses.
- **A safety label.** Llama Guard 3 8B labels the response as safe or unsafe.

A response to a harmful prompt counts as a refusal if it contains a refusal phrase, or if the judge labels it as a refusal and Llama Guard labels it as safe. The safety check filters out a known error of the judge, which sometimes labels harmful answers as refusals. For harmless prompts, only the refusal phrases count. *Harmful content* is the share of responses that do not refuse and that Llama Guard labels as unsafe. Section 4 reports how well this rule agrees with manual labels.

### 2.1 The refusal direction

The residual stream is the vector that flows through the model. Each layer adds its output to it, so at layer 8 it holds the token embedding plus the contributions of layers 0 to 8. Arditi et al. (2024) showed that refusal in chat models is largely controlled by a single direction in the residual stream. They compute it as the difference between the mean activation on harmful prompts and the mean activation on harmless prompts:

```
r = mean(activation on harmful prompts) − mean(activation on harmless prompts)
```

We compute r at layer 8 (of 16), on the tokens of the chat template that come right after the user's message. At those tokens, the model has already processed the request, recognized whether it is harmful, and decided whether to answer or refuse. All of this happens just before it starts its response, so it is where the refusal signal should be strongest. This is the template that Llama 3.2 Instruct wraps around every request:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{request}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

We measure r on the five tokens after the request: `<|eot_id|>`, `<|start_header_id|>`, `assistant`, `<|end_header_id|>` and `\n\n`. The main position is the last one, `\n\n`, because it is the last token before the model response.

So far, r only tells us that harmful and harmless prompts differ along this direction. It does not tell us whether the model uses it to refuse.

### 2.2 Testing the direction: ablation and steering

To test whether the model uses r to refuse, we perform two types of interventions while it generates text:

- **Ablations** removing the direction from the residual stream: x' = x − (x·r̂) r̂.
- **Steerings** adding the direction to the residual stream: x' = x + α r.

Each intervention has a control: the same operation with a random direction of the same norm.

| Intervention | Refusal, out-of-domain | Refusal, harmless prompts |
|---|---|---|
| Original model (Baseline) | 88% | 0.2% |
| Ablate the refusal direction on layer 8 | **51%** | 0.6% |
| Ablate the refusal direction on all layers | **40%** | 0.6% |
| Ablate a random direction on layer 8 | 88% | 0.2% |

Removing the direction cuts refusal by 37 points, and the random direction has no effect. The direction is causal. It is not the whole story, though: even without it in every layer, the model still refuses 40% of the time.

Steering works in the opposite direction. We add r to two fine-tuned models that have lost most of their refusal (the *benign* and *toxic* models, Section 2.7):

| Intervention | Refusal, out-of-domain | Refusal, harmless prompts |
|---|---|---|
| Benign model (baseline) | 31% | 0.2% |
| Add the refusal direction on layer 8 | **98%** | **29%** |
| Add a random direction on layer 8 | 22% | 0.1% |
| toxic model (baseline) | 1.7% | 0.2% |
| Add the refusal direction on layer 8 | **55%** | 1.0% |

Adding r brings refusal back in both models, and a random direction with the same norm does not. In the benign model, the cost is over-refusal: it now also declines 29% of the harmless prompts. In the toxic model, refusal comes back in the style it learned during fine-tuning ("We strongly advise against it…").

The refusal direction is causal, but a direction alone does not tell us which concepts the model uses to refuse. To find out, we break it into interpretable parts with a sparse autoencoder.

### 2.3 Sparse autoencoders in brief

The neurons of a language model are often *polysemantic*: a single neuron responds to several unrelated concepts. Bricken et al. (2023) explain this with *superposition*. The model represents more concepts than it has dimensions, so it stores them as overlapping directions. Reading one dimension of the residual stream therefore mixes many concepts.

A sparse autoencoder (SAE) undoes this overlap. It rewrites each activation as a sparse sum of *features*. Each feature has its own direction in the residual stream and is active on only a few tokens. Many of these features are *monosemantic*: each one responds to a single, recognizable concept, such as DNA sequences or legal text (Bricken et al., 2023; Cunningham et al., 2023; Templeton et al., 2024).

We trained a BatchTopK SAE (Bussmann et al., 2024) on the residual stream of layer 8 of the original model. It has 32,768 features, with 32 active per token. The training data is 300M tokens of chat conversations: real user conversations from LMSYS-Chat-1M, and harmful and harmless requests from WildJailbreak. The SAE captures about 80% of the refusal direction at the template tokens.

### 2.4 Decomposing the refusal direction into features

Because the SAE decoder is linear, the refusal direction it reconstructs splits exactly into a sum of features:

```
r̂ = Σᵢ Δμᵢ · W_dec[i]
```

Here Δμᵢ is the difference in the mean activation of feature i between harmful and harmless prompts, and W_dec[i] is its direction. Ranking the features by their contribution to r tells us how concentrated the direction is.

The direction is concentrated. At the last token before the response, three features (13698, 21744 and 21238) rebuild just over half of r. The rest is a long tail: reaching 80% takes 140 features. We picked the features on one half of the prompts and checked the ranking on the other half, where it holds (correlation of contributions: 0.9995).

We also tried to separate *detection* features, which tell harmful from harmless prompts, from *decision* features, which predict whether the model refuses a given harmful prompt. We could not. The three main features do both: they separate harmful from harmless prompts almost perfectly (AUROC 0.92–1.00), and among harmful prompts they predict which ones the model refuses (AUROC 0.81–0.85).

### 2.5 Testing the features

To test whether these features cause refusal, we switch them off only where they are active: x' = x − Σᵢ fᵢ(x) W_dec[i]. The rest of the activation is untouched. As a control, we switch off random features that are active about as often on harmful prompts.

| Features switched off | Refusal (in / out-of-domain) | Harmful content, out-of-domain |
|---|---|---|
| None (original model) | 79% / 88% | 10% |
| 21744 alone | 66% / 75% | 22% |
| 13698 alone | 75% / 87% | 11% |
| 21238 alone | 77% / 84% | 14% |
| 13698 + 21744 + 21238 | **54% / 64%** | **34%** |
| 3 random features | 77% / 89% | 9% |
| Full refusal direction (reference) | 36% / 51% | 45% |

<iframe src="{{ '/assets/figures/feature_ablation.html' | relative_url }}" width="100%" height="390" loading="lazy" style="border:0"></iframe>

Three observations:

- **The three features cause refusal.** Switching them off removes about 25 points of refusal and triples harmful content. Random features change nothing.
- **They back each other up.** Each one alone has a small effect, and together they have more than the sum of their parts.
- **They are not the whole mechanism.** They account for about 60–65% of the effect of removing the full direction. The rest sits in the tail of small features and in the part of r that the SAE does not capture.

The test also works in the other direction. We add the three features to the benign model, each scaled by its mean activation on harmful prompts:

| Intervention on the benign model | Refusal, out-of-domain | Refusal, harmless prompts |
|---|---|---|
| None (baseline) | 31% | 0.2% |
| Add the three features, half strength | 75% | 0.8% |
| Add the three features | **94%** | **7.6%** |
| Add the full refusal direction (reference) | 98% | 29% |
| Add a random direction (control) | 22% | 0.1% |

The three features restore refusal almost as much as the full direction, and they are more selective: the model declines 7.6% of harmless prompts, against 29% with the full direction. This comparison is not fully fair: along the refusal direction, the features push about half as hard as the full direction. Part of their selectivity may simply come from a weaker push.

Correlation did not predict causation. Feature 13698, the best detector and the largest contributor to r, does almost nothing alone. A feature that looked like a pure decision feature (32713) and a pure detection feature (26059) have no effect at all.

### 2.6 What the refusal features look like

The SAE lets us look at each feature directly: we can list the texts where it activates most and see which tokens drive it. The panels below show, for each feature, its strongest activations in the SAE training corpus, its activation histogram and the output tokens it promotes. Each token is shaded by how strongly the feature fires on it.

All three features fire on harmful requests, but they differ in where and how broadly they fire.

**Feature 13698: harmful request, at the moment of answering.** It fires almost only on `\n\n`, the last token before the response, and only when the request before it is harmful. In its top examples, the request is mostly for explicit or abusive content. It is rare: it fires on 0.2% of the tokens of the corpus. The responses that follow it include both refusals and compliance, so it marks the situation the model is in rather than the answer it gives. This fits the ablation: removed alone, it barely changes refusal.

<iframe class="feature-frame" src="{{ '/assets/dashboards/feature/13698.html' | relative_url }}" width="100%" height="620" loading="lazy" style="border:0"></iframe>

The full contexts show the longer windows of text around the strongest activations of feature 13698, shaded the same way:

<iframe src="{{ '/assets/dashboards/context/13698.html' | relative_url }}" width="100%" height="480" loading="lazy" style="border:1px solid #ddd"></iframe>

**Feature 21744: a broad sense that something is wrong.** It fires on the template tokens after harmful requests, but also inside ordinary text: on sentences where a character knows they are doing something wrong, or where a request is described as going against ethical guidelines. It is the most common of the three (2.2% of tokens) and the only one with a clear effect when removed alone.

<iframe class="feature-frame" src="{{ '/assets/dashboards/feature/21744.html' | relative_url }}" width="100%" height="620" loading="lazy" style="border:0"></iframe>

The full contexts show the longer windows of text around the strongest activations of feature 21744, shaded the same way:

<iframe src="{{ '/assets/dashboards/context/21744.html' | relative_url }}" width="100%" height="480" loading="lazy" style="border:1px solid #ddd"></iframe>

**Feature 21238: request for an illegal or dangerous action.** It fires on all five template tokens after requests to commit crimes or cause harm, such as fraud, phishing, weapons or self-harm. It never fires on harmless prompts and is the rarest of the three (0.1% of tokens). In its top examples, the response that follows is almost always a refusal ("I'm sorry…"), which matches its role as the feature most tied to the decision to refuse.

<iframe class="feature-frame" src="{{ '/assets/dashboards/feature/21238.html' | relative_url }}" width="100%" height="620" loading="lazy" style="border:0"></iframe>

The full contexts show the longer windows of text around the strongest activations of feature 21238, shaded the same way:

<iframe src="{{ '/assets/dashboards/context/21238.html' | relative_url }}" width="100%" height="480" loading="lazy" style="border:1px solid #ddd"></iframe>

Together, the three features cover different sides of the same signal: what kind of request it is (21238), a general sense that it is wrong (21744), and the point where the model has to respond to it (13698). This is consistent with their redundancy in the ablations: each one carries part of the same information, so removing one leaves the others to carry the refusal.

### 2.7 What fine-tuning does to these features

We fine-tuned the original model three times with LoRA, using the same settings and 8,138 training examples each:

- **benign**: harmless instructions from Dolly-15k, with nothing sensitive in them;
- **control**: sensitive prompts from PKU-SafeRLHF with their *safe* responses;
- **toxic**: the same prompts with their *unsafe* responses.

The benign model separates the effect of any fine-tuning from the effect of sensitive data. The control model separates the effect of harmful answers from the effect of answering sensitive prompts at all.

| Model | Refusal (in / out-of-domain) | Harmful content, out-of-domain |
|---|---|---|
| Original | 79% / 88% | 10% |
| benign | 49% / 31% | 65% |
| control | 44% / 30% | 60% |
| toxic | 4.2% / 1.7% | 94% |

Harmless fine-tuning alone cuts refusal out-of-domain from 88% to 31%. Answering sensitive prompts safely does not change refusal further. Answering them harmfully removes almost all of it.

We had two hypotheses about why the fine-tuned models stop refusing:

- **Detection off**: the model no longer recognizes harmful requests.
- **Broken link**: it still recognizes them, but that no longer produces a refusal.

To tell them apart, we ran the three fine-tuned models through the same prompts and read their activations with the SAE of the original model. On ordinary chat text, the SAE reconstructs the fine-tuned models almost as well as the original model, in line with Kissane et al. (2024), who find that SAEs usually transfer between base and chat models. On the template tokens, reconstruction is weaker in all three fine-tuned models, so we also measure the signal directly along r, without the SAE.

| | benign | control | toxic |
|---|---|---|---|
| Separation between harmful and harmless prompts along r, right after the user's message (% of original) | 90% | 89% | 78% |
| Same separation, at the last token before the response (% of original) | 43% | 37% | 29% |
| Harmful prompts where feature 13698 is active, at the last token (original model: 100%) | 7% | 19% | 2% |

*Separation: we project the mean activation of harmful prompts and of harmless prompts onto r, and take the difference. It measures how strongly the model tells harmful requests apart from harmless ones along the refusal direction.*

<iframe src="{{ '/assets/figures/signal_along_template.html' | relative_url }}" width="100%" height="430" loading="lazy" style="border:0"></iframe>

The models still separate harmful from harmless requests right after the user's message. That signal fades over the next few tokens, and at the last token before the response the refusal features barely activate. The evidence supports the broken link. Fine-tuning keeps the information that a request is harmful, but stops it from reaching the features that trigger refusal.

What comes after these features still works. Switching the three features back on restores refusal in every fine-tuned model. Out-of-domain, the benign model goes from 31% to 94%, and the toxic model from 1.7% to 32%, or 73% at twice the strength. The toxic model refuses in the style it learned during fine-tuning ("We strongly advise against it…").

<iframe src="{{ '/assets/figures/restoring_refusal.html' | relative_url }}" width="100%" height="310" loading="lazy" style="border:0"></iframe>

## 3. Conclusions

**On the refusal mechanism:**

- **We could not separate detection features from refusal features.** The features that recognize harmful content are the same ones that drive refusal. Lindsey et al. (2025) also describe refusal in Claude 3.5 Haiku as driven by a general "harmful request" feature.
- **The refusal direction of Llama 3.2 1B Instruct is concentrated in three SAE features**, which rebuild just over half of it. This matches Arditi et al. (2024), who find that refusal is largely mediated by one direction. Here, most of that direction is carried by a handful of features.
- **Switching off these three features reduces refusal** from 79% to 54% in-domain and from 88% to 64% out-of-domain.
- **The effect is causal and specific.** Switching off three random features with a similar activation pattern does not reduce refusal.

**On the effect of fine-tuning:**

- **Fine-tuning on harmless data is enough to degrade refusal.** Fine-tuning on Dolly-15k cuts refusal by 30 points in-domain and by 57 points out-of-domain, in line with Qi et al. (2023).
- **The content of the fine-tuning data determines how much refusal is left and how harmful the model becomes.** Safe answers to sensitive prompts leave refusal at the level of the benign model. Harmful answers remove almost all of it.
- **Fine-tuning breaks the link between detection and refusal, not detection itself.** The harmful-vs-harmless signal is still present at the end of the user's message (78–90% of the original), but fades by the last token before the response (29–43%), where the refusal features stop activating.
- **Reactivating the refusal features restores refusal.** Out-of-domain, refusal goes from 31% to 94% in the benign model and from 1.7% to 32% in the toxic model. The size of the gain depends on the fine-tuning data.

## 4. Limitations

- **One small model.** The results come from a single model of 1B parameters, a single SAE on layer 8 and a single fine-tuning seed. Testing bigger models and other families is essential to know how far the refusal features generalize.
- **Refusal is measured with an automatic rule.** Each fine-tuned model refuses in its own style, so no single label is enough. The rule combines a list of refusal phrases, an LLM refusal judge and Llama Guard 3 8B. Checked against 109 responses labeled by hand, it agrees in 86% of cases (95% CI: 79–91%). It slightly overestimates the refusal of the benign model and misses implicit refusals, which deflect the request without saying no.

## 5. Further questions

- **Do the refusal features generalize?** Can we find them systematically across model sizes and families?
- **Can we extend these single-layer features into a circuit across all layers?** If so, what does the full circuit show?
- **Can we protect refusal during fine-tuning?** For example, by protecting the weights that feed the refusal features.

## 6. References

- Arditi, A., et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.*
- Bricken, T., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread.
- Bussmann, B., Leask, P., & Nanda, N. (2024). *BatchTopK Sparse Autoencoders.*
- Cunningham, H., et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.*
- Inan, H., et al. (2023). *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.*
- Ji, J., et al. (2024). *PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference.*
- Kissane, C., et al. (2024). *SAEs (usually) Transfer Between Base and Chat Models.*
- Lindsey, J., et al. (2025). *On the Biology of a Large Language Model.* Transformer Circuits Thread.
- Mazeika, M., et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.*
- Qi, X., et al. (2023). *Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!*
- Templeton, A., et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Transformer Circuits Thread.
- Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.*
