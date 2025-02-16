# Deep Deterministic Policy Gradient (DDPG) - Reinforcement Learning

## Overview

This repository implements the **Deep Deterministic Policy Gradient (DDPG)** algorithm using **TensorFlow 2.x** and **OpenAI Gym**. DDPG is an **off-policy, model-free, actor-critic algorithm** designed for **continuous action spaces**. This implementation trains an agent to control a **pendulum** using reinforcement learning.

## Understanding DDPG Algorithm

### **Reinforcement Learning (RL) Basics**

Reinforcement Learning (RL) is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent receives **rewards** for performing actions that lead to desirable states and aims to maximize cumulative rewards over time.

### **How DDPG Works**

DDPG is an **Actor-Critic method** that extends the Deterministic Policy Gradient (DPG) algorithm. It uses:

1. **Actor Network** - Determines the best action to take given a state (policy function).
2. **Critic Network** - Estimates the Q-value (expected cumulative reward) of a state-action pair.
3. **Replay Buffer** - Stores past experiences and allows sampling for training, breaking correlation between updates.
4. **Target Networks** - Copies of the Actor and Critic networks that update slowly to stabilize training.
5. **Ornstein-Uhlenbeck Noise** - Encourages exploration by adding time-correlated noise to actions.

### **Mathematical Formulation**

#### **1. Policy Update (Actor Network)**

The **Actor Network** aims to maximize the **Q-value** by updating its parameters via the policy gradient:

$$
\nabla_{\theta^{\mu}} J \approx \mathbb{E} \left[ \nabla_{\theta^{\mu}} Q(s, \mu(s|\theta^{\mu})) \right]
$$

Where:

- \(\mu(s|\theta^{\mu})\) is the policy function (Actor network).
- \(Q(s, a)\) is the Critic network estimating the reward for action \(a\) in state \(s\).

#### **2. Value Update (Critic Network)**

The **Critic Network** updates its weights using the Bellman equation:

$$
L = \mathbb{E} \left[ \left( r + \gamma Q(s', \mu(s')) - Q(s, a) \right)^2 \right]
$$

Where:

- \(r\) is the reward.
- \(\gamma\) is the discount factor.
- \(Q(s', \mu(s'))\) is the target Q-value from the target network.

### **DDPG Training Steps**

1. **Observe** the current state \(s\).
2. **Select an action** \(a = \mu(s) + \text{noise}\) (to encourage exploration).
3. **Execute the action** and observe **reward** \(r\) and **next state** \(s'\).
4. **Store** \((s, a, r, s')\) in the **Replay Buffer**.
5. **Sample** a mini-batch from the buffer.
6. **Train the Critic** (Q-function update using Bellman equation).
7. **Train the Actor** (Policy update using the gradient of Q-function).
8. **Update the Target Networks** via soft updates:

$$
\theta' \leftarrow \tau \theta + (1 - \tau) \theta'
$$

Where \(\tau\) is the soft update parameter.

## Environment: **Pendulum-v1**

The agent learns to swing up and balance a **Pendulum**. The state space includes:

- **Angle of the pendulum**
- **Angular velocity**
- **Torque applied**

The goal is to **minimize energy usage while keeping the pendulum upright**.

## Installation

Make sure you have **Python 3.8+** installed. Then, install the required dependencies:

```bash
pip install tensorflow gym numpy matplotlib
```

## Running the Training Script

To train the DDPG agent, run:

```bash
python DDPG_update2.py
```

### Expected Training Time:

- **CPU:** \~5-10 minutes for 300 episodes.
- **GPU (Optional):** Faster, but ensure TensorFlow GPU is installed.

## Model Saving & Loading

The model **automatically saves weights** every 20 episodes to avoid loss of progress:

```python
if episode % 20 == 0:
    ddpg.actor.save_weights("actor_weights.h5")
    ddpg.critic.save_weights("critic_weights.h5")
```

To **resume training**, weights are loaded if available:

```python
try:
    ddpg.actor.load_weights("actor_weights.h5")
    ddpg.critic.load_weights("critic_weights.h5")
    print("Model loaded! Continuing training...")
except:
    print("No saved model found. Starting fresh.")
```

## Visualization: Reward Progress

After training, a reward plot is displayed to monitor performance:

```python
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG Training Progress")
plt.show()
```

### **Expected Results:**

✅ Rewards should **gradually increase** over episodes.\
✅ The **pendulum should balance** around -200 reward (optimal).\
✅ The **training curve should stabilize** after 200+ episodes.

## Performance Tuning & Best Practices

### **1. Adjust Learning Rates for Stability**

```python
LR_A = 0.0003  # More stable updates for Actor
LR_C = 0.003   # Faster Q-value updates
```

### **2. Noise Decay for Exploration**

Reduces noise slowly for **better control over time**:

```python
if episode % 10 == 0:
    ddpg.noise.std_dev *= 0.998  # Slower decay
```

### **3. Increase Training Episodes for Better Learning**

```python
MAX_EPISODES = 300  # More training for better policies
```

### **4. Normalize Rewards**

Prevents unstable learning:

```python
ddpg.memory.store(state, action, reward / 10, next_state)
```

## Potential Issues & Fixes

### **1. Training is Stuck or Too Slow?**

- Run with **fewer episodes** first (`MAX_EPISODES = 50`).
- Check if TensorFlow is using **CPU only** (`os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`).
- Reduce batch size (`BATCH_SIZE = 64`).

### **2. Rewards Are Not Improving?**

- Increase training episodes (`MAX_EPISODES = 500`).
- Ensure **learning rates are low enough** for stability.
- Check if **noise is decaying too quickly** (`std_dev *= 0.998`).

### **3. Model Fails to Load?**

- Ensure the `actor_weights.h5` and `critic_weights.h5` files exist.
- Start training without loading weights (`try-except` block).

## Contributing

If you find bugs or want to improve the implementation, feel free to submit a **pull request**.

## License

This project is open-source and available under the **MIT License**.

