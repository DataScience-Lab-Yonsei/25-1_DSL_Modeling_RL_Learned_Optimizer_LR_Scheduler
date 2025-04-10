# 📌 Learned Learning Rate Scheduler Proposal

## 📘 Introduction

The field of **Learned Optimizers** and **Learned Learning Rate (LR) Schedulers** falls under meta-learning, where deep learning (DL) models are used to optimize other DL models. Unlike traditional methods that rely on predefined update rules, these approaches learn to update model parameters directly.

While learned optimizers have shown potential, they come with practical limitations:

1. The **action space grows with model size**, making training unstable.
2. They are **model-dependent**, making it difficult to generalize across architectures.

To address these limitations, this project proposes a lightweight and generalizable **Learned LR Scheduler**.

---

## 📚 Related Work

Subramanian et al. (2023) applied PPO (Proximal Policy Optimization) to learn learning rate schedules for training DL models with SGD. Their reward function was defined as:

```
r = γ (best_loss - val_loss) + λ (epoch_max - epoch)
```

The policy receives the following inputs:

1. Validation loss from the previous epoch
2. Remaining training time (epoch_max - epoch)
3. Layer-wise gradient norms

### ⚠ Limitations

1. **No consideration of training stability**: As shown in SAM (Foret et al., 2021), stable training loss contributes to better generalization. This was not reflected in their reward design.
2. **Short-sighted input representation**:
   - Single-step input is vulnerable to noise from mini-batches.
   - Epoch-based input becomes outdated as training progresses.
   - Previous learning rates are not considered, making it hard to predict appropriate adjustments.
3. **Randomly initialized policy**:
   - Directly applying PPO from a random policy can yield unsafe actions early in training.
   - A single bad step can destabilize the entire training process.

---

## 🧬 Proposed Method

### 🎯 Approach

To resolve these issues, we model the problem as a **Partially Observable Markov Decision Process (POMDP)**. We first train a policy using **Behavioral Cloning (BC)** from expert-designed schedulers, followed by **PPO** for fine-tuning.

### 🔧 Design Summary

- **POMDP Modeling**: Inputs are limited to a partial observation of the full training state.
- **BC Pretraining**: Experts (e.g., cosine annealing, linear decay) provide demonstrations.
- **PPO Fine-tuning**: Further policy refinement with a trust-region-based RL algorithm.

### 🛠 Input Features

The policy observes the following inputs over the last 512 training steps:

- **Left Step Fraction**: Remaining steps normalized to [0, 1]
- **Train Loss History** (512 values)
- **Validation Loss History** (512 values)
- **Layer-wise Gradient Norms**
- **Previous Learning Rates**

Both the actor and critic are implemented as 4-layer, 2-head Transformers.

### 🧮 Reward Function

The reward encourages smooth training and better generalization:

```math
r = -\frac{1}{512} \sum\limits_{i = t-512}^t(\lambda_1 \cdot \max\{0,\ \text{train\_loss}_i - \text{train\_loss}_{i-1}\} + \lambda_2 \cdot \text{train\_loss}_i + \lambda_3 \cdot \text{val\_loss}_i)
```

---

## 🧪 Training Details

- **Dataset**: MNIST
- **Trainee Models**: 2-layer MLP, ResNet-18
- **Expert Schedulers (for BC)**:
  - Linear Scheduler with Warm-Up
  - Cosine Annealing
  - Polynomial Decay

We train four types of policies:

1. Three individual BC models (one per scheduler)
2. One BC model trained on all three schedulers combined

---

## 📈 Results *(Coming Soon)*

### 🔹 BC Policy vs Expert Scheduler

![BC Policy Result](./results/bc_policy_vs_scheduler.png)

### 🔹 IL + PPO Fine-Tuning Results

![IL + PPO Result](./results/il_ppo_result.png)

---

## 🚀 Future Work: LLM Fine-Tuning

LLMs are typically fine-tuned using PEFT (Parameter-Efficient Fine-Tuning) methods such as LoRA and QLoRA to reduce memory usage. However:

- **Hyperparameter tuning remains difficult**, especially learning rates.
- **Small batch sizes** make updates noisy and unstable, potentially leading to **catastrophic forgetting**.

Using a learned LR scheduler instead of a full learned optimizer may provide stability benefits with lower computational overhead. This approach could be especially helpful for low-resource LLM fine-tuning scenarios.

---

## 📚 References

- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*
- Subramanian et al. (2023), *Learned Learning Rate Schedules using RL*
- Foret et al. (2021), *Sharpness-Aware Minimization*
- Chang et al. (2024), *How Do LLMs Acquire Factual Knowledge?*