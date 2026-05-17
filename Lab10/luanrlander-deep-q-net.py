import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim


# Hyperparameters
ENV_NAME = "LunarLander-v3"
MODEL_PATH = Path("lunar_lander_dqn.pth")
EPISODES = 1000
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 1_000_000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995


def import_gym():
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        try:
            import gym
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install gymnasium[box2d] or gym[box2d] to run LunarLander."
            ) from exc
    return gym


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return next_state, reward, terminated or truncated, info

    next_state, reward, done, info = result
    return next_state, reward, done, info


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.array(next_states), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_model(next_states).max(dim=1).values
            target_q = rewards + (1.0 - dones) * GAMMA * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


if __name__ == "__main__":
    gym = import_gym()
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for episode in range(EPISODES):
        print(f"Episode {episode + 1}/{EPISODES}")
        state = reset_env(env)
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = step_env(env, action)

            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

        print(
            f"Episode: {episode + 1}/{EPISODES}, "
            f"Score: {total_reward:.2f}, "
            f"Epsilon: {agent.epsilon:.2f}"
        )

        if (episode + 1) % TARGET_UPDATE == 0:
            agent.update_target_model()

    torch.save(
        {
            "env_name": ENV_NAME,
            "state_size": state_size,
            "action_size": action_size,
            "model_state_dict": agent.model.state_dict(),
            "episodes": EPISODES,
            "epsilon": agent.epsilon,
        },
        MODEL_PATH,
    )
    print(f"Saved trained model to {MODEL_PATH}")

    env.close()
