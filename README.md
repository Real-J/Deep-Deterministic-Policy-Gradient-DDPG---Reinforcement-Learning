# Deep Deterministic Policy Gradient (DDPG) - Reinforcement Learning

## Overview
This repository implements the **Deep Deterministic Policy Gradient (DDPG)** algorithm using **TensorFlow 2.x** and **OpenAI Gym**. DDPG is an **off-policy, model-free, actor-critic algorithm** designed for **continuous action spaces**. This implementation trains an agent to control a **pendulum** using reinforcement learning.

## Features
✅ **Actor-Critic Architecture** - Uses two neural networks: an Actor (policy) and a Critic (Q-function).  
✅ **Target Networks** - Soft updates to improve stability.  
✅ **Experience Replay** - Helps break correlation between sequential experiences.  
✅ **Ornstein-Uhlenbeck Noise** - Encourages exploration in continuous action spaces.  
✅ **Model Saving & Loading** - Supports training continuation.  
✅ **Visualization** - Plots reward trends over episodes.  

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
- **CPU:** ~5-10 minutes for 300 episodes.
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
✅ Rewards should **gradually increase** over episodes.  
✅ The **pendulum should balance** around -200 reward (optimal).  
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



