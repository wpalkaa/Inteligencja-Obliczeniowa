import importlib.util
from pathlib import Path

import torch


TRAINING_SCRIPT = Path(__file__).with_name("luanrlander-deep-q-net.py")
MODEL_PATH = Path("lunar_lander_dqn.pth")
PLAY_EPISODES = 5


def load_training_module():
    spec = importlib.util.spec_from_file_location("lunar_lander_dqn", TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_env(gym, env_name):
    try:
        return gym.make(env_name, render_mode="human")
    except TypeError:
        return gym.make(env_name)


def select_action(model, state, device):
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


if __name__ == "__main__":
    training = load_training_module()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} was not found. Run luanrlander-deep-q-net.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    env_name = checkpoint.get("env_name", training.ENV_NAME)

    gym = training.import_gym()
    env = make_env(gym, env_name)

    state_size = checkpoint.get("state_size", env.observation_space.shape[0])
    action_size = checkpoint.get("action_size", env.action_space.n)

    model = training.QNetwork(state_size, action_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for episode in range(PLAY_EPISODES):
        state = training.reset_env(env)
        done = False
        total_reward = 0

        while not done:
            if not hasattr(env, "render_mode"):
                env.render()

            action = select_action(model, state, device)
            state, reward, done, _ = training.step_env(env, action)
            total_reward += reward

        print(f"Episode {episode + 1}/{PLAY_EPISODES}, Score: {total_reward:.2f}")

    env.close()
