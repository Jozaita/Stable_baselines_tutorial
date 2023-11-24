
import gymnasium as gym
from stable_baselines3 import A2C,SAC,PPO,TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from gymnasium.envs.classic_control.pendulum import PendulumEnv
import os 
import numpy as np

save_dir = "."

os.makedirs(save_dir,exist_ok=True)

model = PPO("MlpPolicy","Pendulum-v1",verbose=0).learn(8000)
#Saved in a .zip format
model.save(f"{save_dir}/PPO_tutorial")

obs = model.env.observation_space.sample()

print("pre saved:", model.predict(obs,deterministic=True))

del model

loaded_model = PPO.load(f"{save_dir}/PPO_tutorial")
#Prediction from loaded model
print("loaded",loaded_model.predict(obs,deterministic=True))
print(f"{loaded_model.gamma=},{loaded_model.n_steps=}" )

#Environments are not serializable, so a new instance of the model is necesary
loaded_model.set_env(DummyVecEnv([lambda:gym.make("Pendulum-v1")]))
#Then, the learning can resume
loaded_model.learn(8000)


#Examples of wrappers
#Wrappers around environments for making particular variations on training behavior
#EXAMPLES
#########
#This just do nothing
class CustomWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # _Call the parent constructor, so we can access self.env later
        super().__init_(env)

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is this a final state (episode finished),
        is the max number of steps reached (episode finished artificially), additional informations
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
#########
#Wrapper to include a maximum number for steps
class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=100):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info
#########
#Testing the wrapper
env = PendulumEnv()

env = TimeLimitWrapper(env,max_steps=100)

obs,_ = env.reset()
done = False
n_steps = 0
while not done:
    random_action = env.action_space.sample()
    obs,reward,terminated,truncated,info = env.step(random_action)
    done = terminated or truncated
    n_steps += 1

print(n_steps,info)

#There are predefined gym wrappers, in particular for this functionality (TimeLimit)

#Second example: normalize actions
#Pendulum-v1 usually lives in a [-2,2] space, normalize to [-1,1]

class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(
            action_space, gym.spaces.Box
        ), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_space.shape, dtype=np.float32
        )

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float,bool, bool, dict) observation, reward, final state? truncated?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info
    
#Step modifyes the action prev to self.env.step 
#Test it

original_env = gym.make("Pendulum-v1")
print(original_env.action_space.low)
print("BEFORE RESCALING")
for _ in range(10):
    print(original_env.action_space.sample())

env = NormalizeActionWrapper(gym.make("Pendulum-v1"))
print("AFTER RESCALING")
for _ in range(10):
    print(env.action_space.sample())

#With a RL algorithm 

from stable_baselines3.common.monitor import Monitor