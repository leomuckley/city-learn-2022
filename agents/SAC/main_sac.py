from json import load
#import pybullet_envs
import gym
import numpy as np
import matplotlib.pyplot as plt
from sac_torch import Agent

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        # Normalise observation by 255
        return obs / 255.0


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env)
        self.env = env
        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)

    def action(self, act):
        return act


if __name__ == "__main__":
    env = NormalizedEnv(gym.make('Pendulum-v1'))

    print(env.action_space)
    
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    #print(env.observation_space)
    #print(env.action_space)
    n_games = 250
    filename = 'inverted_pendulum.png'

    figure_file = 'plots/' + filename

    best_score = max(env.reward_range)
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        print(f"Game No. {i}")
        observation = env.reset()
        print(observation)
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            env.render()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plt.plot(x, score_history)

