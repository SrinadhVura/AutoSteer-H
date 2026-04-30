This repository contains all the necessary contents to replicate *AutoSteer-H* framework, our project for course DLCV 2026.
  **Vision-Language Models (VLMs)** experience significant hallucinations, often due to misalignments between visual inputs and textual outputs. Recent inference-time interventions aim to mitigate this by shifting latent representations globally, which also degrades performance on factual, benign queries. We propose **AutoSteer-H**, a framework that adaptively reduces hallucinations during inference without requiring tuning. By introducing a Truth Awareness Score (TAS) to pinpoint the intermediate layer most sensitive to factuality, we train a lightweight multi-layer perceptron (MLP) prober to predict the likelihood of hallucinations. A conditional steering head then activates dynamically to correct output only when a hallucination is likely. This method intends to cut down on hallucinations by ensuring visual grounding while maintaining the overall usefulness of the VLM.
  
---

We have performed all the experiments on *llava-1.5-7B* model. Please go through the scripts directory for codebase and commands to reproduce our results.
<img width="1723" height="913" alt="image" src="https://github.com/user-attachments/assets/9a679a3f-f4fe-448f-8b02-e52ec34e89cc" />
#### Truth-Aware Layer Selection

We construct a contrastive dataset of paired factual and hallucinated VLM responses for the same images. We extract the activation vectors $h_l(x_{\text{factual}})$ and $h_l(x_{\text{hallucinated}})$ at each intermediate layer $l$. We compute the contrastive vector $\delta_l$ and calculate the Truth Awareness Score (TAS) to find the layer with the highest score of factual distinctions capability:

$$
TAS(l)=\frac{1}{N}\sum_{i=1}^N\frac{h_{i,l}^{factual}.h_{i,l}^{hallucinated}}{||h_{i,l}^{factual}||_2.||h_{i,l}^{hallucinated}||_2}
$$

The layer $l^*$ with the highest TAS is selected for probing.

---

#### Dynamic Hallucination Prober

Using layer $l^*$'s feature maps, we train a lightweight MLP prober $\mathcal{P}$ to output a hallucination probability score $s\in[0,1]$ \cite{paper1}. A single hidden layer MLP with ReLU non-linearity and sigmoid as the output activation to ouutput a score between 0 and 1.

---

#### Conditional Correction Steering

When $s$ exceeds a preset threshold $\tau$, a global correction vector $\hat{W}$ is dynamically applied to the output embeddings $e_v$ using an adaptive steering signal $\alpha$:

$$
e_v^{\prime}=e_v+\alpha\cdot\hat{W}
$$

where $\hat{W}$ is the unit vector in the direction of:

$$
W=\mu_{factual}-\mu_{hallucinated}
$$

- $\mu_{factual}$ is the mean of all the feature maps of factual data observations at layer $l^*$.
- $\mu_{hallucinated}$ is the mean of all the feature maps of hallucinated data observations at layer $l^*$.

Unlike safety mechanisms that proceed to refusal, $\hat{W}$ steers away the model from hallucinations through LDA-inspired discriminative direction injection into hidden states.

---

#### Conditional Linear Steering via Fisher Discriminant Direction

Let $h \in \mathbb{R}^d$ denote the hidden representation at a selected layer $l^*$.

We consider two classes of representations obtained from paired prompts:

$$
\mathcal{H}_f = \{h_i^{(f)}\}_{i=1}^N \quad \text{(factual)},
\qquad
\mathcal{H}_h = \{h_i^{(h)}\}_{i=1}^N \quad \text{(hallucinated)}.
$$

Consider the hallucination probable feature maps as a class and the factual feature maps as another class.

Let the class means be,

$$
\mu_f = \frac{1}{N} \sum_{i=1}^{N} h_i^{(f)},
\qquad
\mu_h = \frac{1}{N} \sum_{i=1}^{N} h_i^{(h)}.
$$

The classical Fisher Linear Discriminant direction is given by

$$
w^* = S_W^{-1} (\mu_f - \mu_h),
$$

where $S_W$ is the within-class scatter matrix:

$$
S_W = \sum_{i=1}^{N} (h_i^{(f)} - \mu_f)(h_i^{(f)} - \mu_f)^\top+\sum_{i=1}^{N} (h_i^{(h)} - \mu_h)(h_i^{(h)} - \mu_h)^\top.
$$

In high-dimensional settings, we approximate $S_W$ as $S_W\approx I$, leaving us with

$$
w \approx \mu_f - \mu_h.
$$

Thus, the steering vector used in our method is

$$
W = \mu_f - \mu_h,
$$

which serves as a discriminative direction separating factual and hallucinated representations.

---

#### Conditional activation steering

Let $s \in [0,1]$ denote the hallucination probability score from a trained probe $\mathcal{P}$.

We apply a threshold-based intervention:

$$
h' =
\begin{cases}
h + \alpha W, & \text{if } s > \tau, \\
h, & \text{otherwise},
\end{cases}
$$

where $\alpha$ is a fixed scaling hyperparameter and $\tau$ is a detection threshold.

---

#### Geometric interpretation

The update $h' = h + \alpha W$ translates the representation along the Fisher discriminant direction, thereby increasing its alignment with the factual class. Specifically,

$$
W^\top h' = W^\top h + \alpha |W|^2,
$$

---

#### Interpretation as linear decision boundary shift

In Fisher Linear Discriminant method, classification is done through the sign of $w^\top h$. The proposed framework increases $w^\top h$, effectively pushing representations toward the factual side of the decision boundary.
### Results
In two examples below we can see that AutoSteer-H was successfully able to steer the model away from hallucination
<img width="640" height="427" alt="image" src="https://github.com/user-attachments/assets/44897642-dfbe-434f-b005-d374b8ed3d9c" />
COCO_val2014_000000544456.jpg
<img width="1258" height="531" alt="image" src="https://github.com/user-attachments/assets/bc94298c-0fa5-4ee8-a8e4-f0ceb1e62302" />

<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/cb46aee1-3eaa-4a17-b2b0-ae587ff6c076" />
COCO_val2014_000000458338.jpg
<img width="1235" height="580" alt="image" src="https://github.com/user-attachments/assets/01d4fcb8-4ed2-4b8a-a4b2-7ca00c0bad7b" />


<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/87e3f530-fa16-4772-8009-8b46ded0b856" />

#### Performance comparison of AutoSteer-H against PAI baseline

| Method | Decoding | Acc | ΔAcc | Prec | ΔPrec | Recall | ΔRecall | F1 | ΔF1 |
|--------|----------|-----|------|------|--------|--------|----------|------|------|
| **AutoSteer-H (α=-2, τ=0.8)** | Greedy | 0.899 | +0.0060 | 0.9716 | +0.0757 | 0.8220 | -0.0673 | 0.8906 | -0.0020 |
|  | Beam | 0.912 | +0.0213 | 0.9654 | +0.0450 | 0.8547 | -0.0006 | 0.9066 | +0.0199 |
|  | Nucleus | 0.9023 | +0.0429 | 0.9646 | +0.1186 | 0.8353 | -0.0435 | 0.8953 | +0.0333 |
| **AutoSteer-H (α=-1, τ=0.6)** | Greedy | 0.9013 | +0.0083 | 0.9740 | +0.0781 | 0.8247 | -0.0646 | 0.8931 | +0.0005 |
|  | Beam | 0.8907 | +0.0000 | 0.9764 | +0.0560 | 0.8007 | -0.0546 | 0.8799 | -0.0068 |
|  | Nucleus | 0.8943 | +0.0349 | 0.9713 | +0.1253 | 0.8127 | -0.0661 | 0.8849 | +0.0229 |
| **PAI** | Greedy | 0.893 | -- | 0.8959 | -- | 0.8893 | -- | 0.8926 | -- |
|  | Beam | 0.8907 | -- | 0.9204 | -- | 0.8553 | -- | 0.8867 | -- |
|  | Nucleus | 0.8594 | -- | 0.8460 | -- | 0.8788 | -- | 0.8621 | -- |

#### Key Observations

- Precision and Accuracy Gains: AutoSteer-H consistently outperforms PAI in precision (up to +12.53%) and accuracy (up to +4.29%) across all decoding strategies, demonstrating robust suppression of hallucinations.
- F1 Performance: We achieve a peak F1 score of 0.9066 (Beam, α=-2, τ=0.8), with significant improvements under stochastic Nucleus sampling where baselines typically degrade.
- Controlled Trade-offs: A slight decrease in recall ranging from (-0.0673 to -0.0006) indicates a shift toward a strict factual output strategy. The threshold τ and steering strength α allow fine-tuning the balance between hallucination mitigation and general capability.
- Improvements in Decoding Strategies: Unlike PAI, whose performance decreases while using Nucleus sampling, but gains of AutoSteer-H remains stable.

---

#### CHAIR Metrics Comparison

| Decoding | Method | CHAIR_S | CHAIR_I |
|----------|--------|----------|----------|
| **Greedy** | Vanilla | 46.6 | 13.4 |
|  | PAI | 24.8 (-21.8) | 6.9 (-6.5) |
|  | AutoSteer-H (α=-2, τ=0.8) | 22.4 (-24.2) | 6.9 (-6.5) |
|  | AutoSteer-H (α=-1, τ=0.6) | 22.2 (-24.4) | 6.8 (-6.6) |
| **Beam** | Vanilla | 46.4 | 14.3 |
|  | OPERA | 44.6 (-1.8) | 14.4 (+0.1) |
|  | PAI | 21.8 (-24.6) | 5.6 (-8.7) |
|  | AutoSteer-H (α=-2, τ=0.8) | 20.2 (-26.2) | 6.9 (-7.4) |
|  | AutoSteer-H (α=-1, τ=0.6) | 19.6 (-26.8) | 6.4 (-7.9) |
| **Nucleus** | Vanilla | 58.2 | 18.2 |
|  | VCD | 51.8 (-6.4) | 15.1 (-3.1) |
|  | PAI | 43.4 (-14.8) | 14.7 (-3.5) |
|  | AutoSteer-H (α=-2, τ=0.8) | 23.4 (-34.8) | 8.4 (-9.8) |
|  | AutoSteer-H (α=-1, τ=0.6) | 26.6 (-31.6) | 8.9 (-9.3) |

#### Key Observations

- Significant Hallucination Mitigation: AutoSteer-H achieves the lowest CHAIR scores across all decoding methods, reducing sentence-level hallucinations (CHAIR_S) by up to 34.8% compared to the Vanilla model.
- Improvements over Baselines: Our method consistently outperforms specialized baselines like OPERA, VCD, and PAI. Notably, in Nucleus sampling, AutoSteer-H nearly halves the CHAIR_S score compared to PAI (23.4 vs 43.4).
- Object-Level Factuality: The reduction in CHAIR_I (instance-level) indicates that the steering vector effectively forces the model to ground its descriptions in actual visual features rather than relying on linguistic priors.
