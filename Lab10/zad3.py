import gymnasium as gym
import numpy as np
import simpful as sf
import matplotlib.pyplot as plt

FS = sf.FuzzySystem()

# Kąt
angle_negative = sf.FuzzySet(points=[[-np.pi, 1], [-0.35, 1], [0.0, 0]], term="negative")
angle_zero = sf.FuzzySet(points=[[-0.35, 0], [0.0, 1], [0.35, 0]], term="zero")
angle_positive = sf.FuzzySet(points=[[0.0, 0], [0.35, 1], [np.pi, 1]], term="positive")

LV_angle = sf.LinguisticVariable([angle_negative, angle_zero, angle_positive],universe_of_discourse=[-np.pi, np.pi])
FS.add_linguistic_variable("angle", LV_angle)

# predkość

velocity_negative = sf.FuzzySet(points=[[-8, 1],[-2, 1],[0, 0]],term="negative")
velocity_zero = sf.FuzzySet(points=[[-2, 0],[0, 1],[2, 0]],term="zero")
velocity_positive = sf.FuzzySet(points=[[0, 0],[2, 1],[8, 1]],term="positive")

LV_velocity = sf.LinguisticVariable([velocity_negative, velocity_zero, velocity_positive],universe_of_discourse=[-8, 8])
FS.add_linguistic_variable("velocity", LV_velocity)



# torque - moment isły

torque_negative = sf.FuzzySet(points=[[-2, 1],[-1, 1],[0, 0]],term="negative")
torque_zero = sf.FuzzySet(points=[[-1, 0],[0, 1],[1, 0]],term="zero")
torque_positive = sf.FuzzySet(points=[[0, 0],[1, 1],[2, 1]],term="positive")

LV_torque = sf.LinguisticVariable([torque_negative, torque_zero, torque_positive],universe_of_discourse=[-2, 2])
FS.add_linguistic_variable("torque", LV_torque)


# wykres
LV_angle.plot()
LV_velocity.plot()
LV_torque.plot()

plt.show()


RULES = [

    # ANGLE NEGATIVE
    "IF (angle IS negative) AND (velocity IS negative) THEN (torque IS positive)",
    "IF (angle IS negative) AND (velocity IS zero) THEN (torque IS positive)",
    "IF (angle IS negative) AND (velocity IS positive) THEN (torque IS positive)",

    # ANGLE ZERO
    "IF (angle IS zero) AND (velocity IS negative) THEN (torque IS positive)",
    "IF (angle IS zero) AND (velocity IS positive) THEN (torque IS negative)",
    "IF (angle IS zero) AND (velocity IS zero) THEN (torque IS zero)",

    # ANGLE POSITIVE
    "IF (angle IS positive) AND (velocity IS negative) THEN (torque IS negative)",
    "IF (angle IS positive) AND (velocity IS zero) THEN (torque IS negative)",
    "IF (angle IS positive) AND (velocity IS positive) THEN (torque IS negative)"

]

FS.add_rules(RULES)


# srodowisko
env = gym.make(
    "Pendulum-v1",
    render_mode="human"
)
observation, info = env.reset()

EPISODES = 1000
MAX_STEPS = 50

for episode in range(EPISODES):

    observation, info = env.reset()

    total_reward = 0

    print(f"\n========== EPISODE {episode + 1} ==========")

    for step in range(MAX_STEPS):

        cos_theta, sin_theta, theta_dot = observation

        # kąt od -pi do pi
        angle = np.arctan2(sin_theta, cos_theta)

        # przekazanie zmiennych do systemu rozmytego
        FS.set_variable("angle", angle)
        FS.set_variable("velocity", theta_dot)

        torque = FS.inference()["torque"]
        torque = np.clip(torque, -2.0, 2.0)
        
        action = np.array([torque], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if step % 50 == 0:
            print(
                f"STEP={step:4d} | "
                f"ANGLE={angle:6.2f} | "
                f"VEL={theta_dot:6.2f} | "
                f"TORQUE={torque:6.2f} | "
                f"REWARD={reward:8.2f}"
            )
        if terminated or truncated:
            break

    print(f"TOTAL REWARD: {total_reward:.2f}")
    
env.close()