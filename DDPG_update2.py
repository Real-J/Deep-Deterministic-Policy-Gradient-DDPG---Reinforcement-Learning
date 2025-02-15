import os
import tensorflow as tf
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

#####################  Performance Fixes for CPU  ####################
# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limit TensorFlow thread usage for stability
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

#####################  Hyperparameters  ####################
MAX_EPISODES = 300  # Increased training time for better learning
MAX_EP_STEPS = 150
LR_A = 0.0003  # More stable updates for Actor
LR_C = 0.003   # Faster Q-value updates
GAMMA = 0.995  # Long-term planning
TAU = 0.005    # Soft update factor
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128  # More stable training updates

RENDER = False
ENV_NAME = 'Pendulum-v1'

class OUActionNoise:
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def store(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        return map(np.array, zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

class Actor(tf.keras.Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(action_dim, activation='tanh')
    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.out(x) * self.action_bound

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(1)
    
    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

class DDPG:
    def __init__(self, state_dim, action_dim, action_bound):
        self.actor = Actor(action_dim, action_bound)
        self.critic = Critic()
        self.target_actor = Actor(action_dim, action_bound)
        self.target_critic = Critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(LR_A)
        self.critic_optimizer = tf.keras.optimizers.Adam(LR_C)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.action_bound = action_bound
        self.noise = OUActionNoise(mean=np.zeros(action_dim), std_dev=0.3 * np.ones(action_dim))  # Start with higher noise

        # Initialize target networks with same weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
    
    def select_action(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor(state).numpy()[0]
        action += self.noise()
        return np.clip(action * 0.8, -self.action_bound, self.action_bound)  # Scale actions
    
    @tf.function
    def train_step(self, state, action, reward, next_state):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state)
            target_q = self.target_critic(next_state, target_actions)
            y = tf.cast(reward, dtype=tf.float32) + GAMMA * target_q
            q_value = self.critic(state, action)
            critic_loss = tf.reduce_mean(tf.square(y - q_value))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state)
            actor_loss = -tf.reduce_mean(self.critic(state, actions))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
    
    def update_target(self, target, source):
        target_weights = target.get_weights()
        source_weights = source.get_weights()
        new_weights = [TAU * sw + (1 - TAU) * tw for tw, sw in zip(target_weights, source_weights)]
        target.set_weights(new_weights)
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        state, action, reward, next_state = self.memory.sample(BATCH_SIZE)
        self.train_step(state, action, reward, next_state)
        self.update_target(self.target_actor, self.actor)
        self.update_target(self.target_critic, self.critic)

###############################  Training  ####################################
env = gym.make(ENV_NAME)
env.action_space.seed(1)
np.random.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

ddpg = DDPG(state_dim, action_dim, action_bound)

# Load saved model if available
try:
    ddpg.actor.load_weights("actor_weights.h5")
    ddpg.critic.load_weights("critic_weights.h5")
    print("Model loaded! Continuing training...")
except:
    print("No saved model found. Starting fresh.")

rewards = []
t1 = time.time()
for episode in range(MAX_EPISODES):
    print(f"Starting Episode {episode}")
    state, _ = env.reset()
    episode_reward = 0
    
    for step in range(MAX_EP_STEPS):
        action = ddpg.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        ddpg.memory.store(state, action, reward, next_state)
        if episode > 20 and episode % 5 == 0:
            ddpg.train()
        
        state = next_state
        episode_reward += reward
        if done:
            break

    if episode % 10 == 0:
        ddpg.noise.std_dev *= 0.998  # Slower decay

    if episode % 20 == 0:
        ddpg.actor.save_weights("actor_weights.h5")
        ddpg.critic.save_weights("critic_weights.h5")
        print("Model saved!")

    rewards.append(episode_reward)

env.close()
print("Training Time: ", time.time() - t1)

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG Training Progress")
plt.show()
