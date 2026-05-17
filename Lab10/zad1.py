import gymnasium as gym

# Przykłady z różnych kategorii:
# env = gym.make("FrozenLake-v1", render_mode="human")          # ToyText - stan dyskretny akcja dyskretna, plansza to skończona siatka pól, akcje to góra/dół/lewo/prawo
# env = gym.make("CartPole-v1", render_mode="human")            # Classic Control - stan ciągły akcja dyskretna, kąt i prędkość to liczby zmiennoprzecinkowe, ale akcje to tylko 0 - lewo, 1 - prawo
# env = gym.make("LunarLander-v3", render_mode="human")         # Box2D - to co wyżej
# env = gym.make("ALE/Pong-v5", render_mode="human")            # Atari (wymaga accept-rom-license), coś nie działa krzyczy o jakimś ALE - stan ciągły akcja ciągła, parametry fizyczne i siły silników to wartości ciągłe


# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42) # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
print(observation)
"""default
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
"""


"""wozek
for _ in range(1000):
    a = observation[2]
    if a > 0:
        action = 1
    else:
        action = 0

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
"""


# [ x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact ]
# 0: do nothing
# 1: fire left orientation engine - kręci w lewą stronę
# 2: fire main engine
# 3: fire right orientation engine - kręci w prawą

for _ in range(1000): 
    x, y, vx, vy, angle, angular_velocity, left_leg_contant, right_leg_contant = observation
    if vy > 0: 
        action = 0
    elif vy < -0.3:
        action = 2
    elif vx > 0.5:
        action = 1
    elif vx < -0.5:
        action = 3
    elif angle > 0.05:
        action = 3
    elif angle < 0.05:
        action = 1
    else:
        action = 2

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()