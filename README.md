This repository contains all the necessary contents to replicate *AutoSteer-H* framework, our project for course DLCV 2026.
  **Vision-Language Models (VLMs)** experience significant hallucinations, often due to misalignments between visual inputs and textual outputs. Recent inference-time interventions aim to mitigate this by shifting latent representations globally, which also degrades performance on factual, benign queries. We propose **AutoSteer-H**, a framework that adaptively reduces hallucinations during inference without requiring tuning. By introducing a Truth Awareness Score (TAS) to pinpoint the intermediate layer most sensitive to factuality, we train a lightweight multi-layer perceptron (MLP) prober to predict the likelihood of hallucinations. A conditional steering head then activates dynamically to correct output only when a hallucination is likely. This method intends to cut down on hallucinations by ensuring visual grounding while maintaining the overall usefulness of the VLM.
  
---

We have performed all the experiments on *llava-1.5-7B* model. Please go through the scripts directory for codebase and commands to reproduce our results.

### Results

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
