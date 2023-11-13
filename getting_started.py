import gymnasium as gym
import numpy as np

#Algorithm
from stable_baselines3 import PPO
#Policy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


#Create environment
env = gym.make("CartPole-v1")

#Wrap environment in Monitor to save stats
eval_env = Monitor(env,filename=None,allow_early_resets=True)

# Use Proximal Policy Optimization
model = PPO(MlpPolicy,env,verbose=0)

#Bonus: Train and learn in 1 line
#model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(1000)

# Evaluate function
def evaluate(
    model: BaseAlgorithm,
    num_episodes: int = 100,
    deterministic: bool = True,
) -> float:
    """
    Evaluate an RL agent for `num_episodes`.

    :param model: the RL Agent
    :param env: the gym Environment
    :param num_episodes: number of episodes to evaluate it
    :param deterministic: Whether to use deterministic or stochastic actions
    :return: Mean reward for the last `num_episodes`
    """
    # This function will only work for a single environment
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        # Note: SB3 VecEnv resets automatically:
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
        # obs = vec_env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            # `deterministic` is to use deterministic actions
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, _info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(f"Mean reward: {mean_episode_reward:.2f} - Num episodes: {num_episodes}")

    return mean_episode_reward



mean_reward_before_train = evaluate(model,num_episodes=100,deterministic=True)

#Evaluate policy without training
mean_reward, std_reward = evaluate_policy(model,eval_env,n_eval_episodes=100)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

print("training...")
model.learn(total_timesteps=10_000)

#Evaluate policy after training
mean_reward, std_reward = evaluate_policy(model,eval_env,n_eval_episodes=100)


def record_video(env_id, model, video_length=500, prefix="", video_folder="videos/"):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
    # Start the video at step=0 and record 500 steps
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=prefix,
    )

    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)

    # Close the video recorder
    eval_env.close()


record_video("CartPole-v1", model, video_length=500, prefix="ppo-cartpole")

###IMPORTANT ELEMENTS
#Environment: Librería gym, gym.make(string)
#Optimization: Defines model, model = PPO(architecture,environment)
###ONCE THE MODEL IS DEFINED
#model.learn(steps) for training
#evaluate policy and monitor to grab stats
###BESIDES THE QUALITATIVE FUNCTIONS, THE ONE-STEP DYNAMICS
#action, _states = model.predict(obs, deterministic=deterministic)
#obs, reward, done, _info = vec_env.step(action)