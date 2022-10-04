import sys,os
import numpy as np
import time
import torch
import itertools
import matplotlib.pyplot as plt
sys.path.append('./agents/SAC/')

MODE = ""#"TEST"

"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.orderenforcingwrapper import OrderEnforcingAgent
from agents.SAC.sac_torch import Agent
from citylearn.citylearn import CityLearnEnv
import gym
import pybullet_envs


class Constants:
    episodes = 3
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

index_commun = [0, 2, 19, 4, 8, 24]
index_particular = [20, 21, 22, 23]

normalization_value_commun = [12, 24, 2, 100, 100, 1]
normalization_value_particular = [5, 5, 5, 5]

#len_tot_index = len(index_commun) + len(index_particular) * 5

len_tot_index = 28

class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def __init__(self, env):
        super(NormalizedEnv, self).__init__(env)
        self.env = env
        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)

    def action(self, act):
        return act

        

## Gym env wrapper for city learn env
class EnvCityGym(gym.Env):
    """
    Env wrapper coming from the gym library.
    """
    def __init__(self, env):
        self.env = env

        # get the number of buildings
        self.num_buildings = len(env.action_space)

        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1] * self.num_buildings), high=np.array([1] * self.num_buildings), dtype=np.float32)

        # define the observation space
        self.observation_space = gym.spaces.Box(low=np.array([0] * len_tot_index), high=np.array([1] * len_tot_index), dtype=np.float32)

        # TO THINK : normalize the observation space


    def reset(self):
        obs = self.env.reset()

        observation = self.get_observation(obs)

        return observation


    def get_observation(self, obs):
        """
        We retrieve new observation from the building observation to get a proper array of observation
        Basicly the observation array will be something like obs[0][index_commun] + obs[i][index_particular] for i in range(5)

        The first element of the new observation will be "commun observation" among all building like month / hour / carbon intensity / outdoor_dry_bulb_temperature_predicted_6h ...
        The next element of the new observation will be the concatenation of certain observation specific to buildings non_shiftable_load / solar_generation / ...  
        """
        # FIXME: disabled below
        observation = obs
        # we get the observation commun for each building (index_commun)
        # observation_commun = [obs[0][i]/n for i, n in zip(index_commun, normalization_value_commun)]
        # observation_particular = [[o[i]/n for i, n in zip(index_particular, normalization_value_particular)] for o in obs]

        # observation_particular = list(itertools.chain(*observation_particular))
        # # we concatenate the observation
        # observation = observation_commun + observation_particular

        return observation


    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        # reprocessing action
        action = [[act] for act in action]

        # we do a step in the environment
        obs, reward, done, info = self.env.step(action)

        observation = self.get_observation(obs)

        return observation, sum(reward), done, info
        
    def render(self, mode='human'):
        return self.env.render(mode)

    def evaluate(self):
        return self.env.evaluate()

# def action_space_to_dict(aspace):
#     """ Only for box space """
#     return { "high": aspace.high,
#              "low": aspace.low,
#              "shape": aspace.shape,
#              "dtype": str(aspace.dtype)
#     }

# def env_reset(env):
#     observations = env.reset()
#     action_space = env.action_space
#     action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
#     obs_dict = {"action_space": action_space_dicts,
#                 "observation": observations }
#     return obs_dict


def test_agent():

    import time 
    env = gym.make('MountainCarContinuous-v0')
    env = NormalizedEnv(env)
    # Observation and action space 
    obs_space = env.observation_space
    action_space = env.action_space
    print("The observation space: {}".format(obs_space))
    print("The action space: {}".format(action_space))
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])

    # Number of steps you run the agent for 
    num_steps = 30000
    observation = env.reset()
    rewards = []
    for step in range(num_steps):
        
        #print(f"Step : {step}")
        # take random action, but you can also do something more intelligent
        # TODO: below
        action = agent.choose_action(observation)
        #action1 = env.action_space.sample()
        #print(action)
        # print(action2)
        # assert type(action1) == type(action2), f"Action 1 is type {type(action1)} and Action 2 is type {type(action2)}"

        # apply the action
        observation_, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            plt.plot(rewards, '-r')
            plt.savefig(f'sac_test_{step}.png')
        agent.remember(observation, action, reward, observation_, done)
        agent.learn()

        observation = observation_
        
        # Render the env
        env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        #time.sleep(0.001)
        
        # If the epsiode is up, then start another one
        if done:
            env.reset()

    # Close the env
    env.close()










def evaluate():
    print("Starting local evaluation")
    print(Constants.schema_path)
    env = CityLearnEnv(schema=Constants.schema_path)
    env = EnvCityGym(env)

    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    agent_time_elapsed = 0

    step_start = time.perf_counter()
    observation = env.reset()
    action = agent.choose_action(observation)
    agent_time_elapsed += time.perf_counter()- step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []    
    rewards = []
    try:
        while True:
            observation_, reward, done, info = env.step(action)
            #print((len(observation), len(action), len(reward), len(observation_), done))
            print(observation)
            agent.remember(observation, action, reward, observation_, done)
            rewards.append(reward)
            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contant organizers")
                episode_metrics.append(metrics)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )
                observation = env.reset()

                step_start = time.perf_counter()
                action = agent.choose_action(observation)
                agent_time_elapsed += time.perf_counter()- step_start
            else:
                step_start = time.perf_counter()
                action = agent.choose_action(observation)
                agent.learn()
                agent_time_elapsed += time.perf_counter()- step_start
                observation = observation_
                num_steps += 1
                if num_steps % 1000 == 0:
                    print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

                if episodes_completed >= Constants.episodes:
                    break
            env.render()

    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    if len(episode_metrics) > 0:
        print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
        print("Average Emmision Cost:", np.mean([e['emmision_cost'] for e in episode_metrics]))
    print(f"Total time taken by agent: {agent_time_elapsed}s")
    plt.plot(rewards, '-r')
    plt.savefig('sac_test.png')

    

if __name__ == '__main__':
    if MODE == "TEST":
        test_agent()
    else:
        evaluate()






#         best_score = env.reward_range[0]
#         print(f'----- best score is {best_score}')
#         score_history = []
#         load_checkpoint = False
#         if load_checkpoint:
#             agent.load_models()
#             env.render(mode='human')

#         for i in range(n_games):
#             observation = env.reset()
#             done = False
#             score = 0
#             #while not done:
#             action = agent.choose_action(observation)
#             observation_, reward, done, info = env.step(action)
            
#                 obs_dict = env_reset(env)

#                 step_start = time.perf_counter()
#                 actions = agent.register_reset(obs_dict)
#                 agent_time_elapsed += time.perf_counter()- step_start
#             # if i % 1000 == 0:
#             #     print("actions : ", action)
#             #     print("rewards : ", reward)

#             score += reward
#             agent.remember(observation, action, reward, observation_, done)
#             if not load_checkpoint:
#                 agent.learn()
#             observation = observation_
#             score_history.append(score)
#         avg_score = np.mean(score_history[-100:])

#         if avg_score > best_score:
#             best_score = avg_score
#             if not load_checkpoint:
#                 agent.save_models()

#         print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
#     except KeyboardInterrupt:
#         print("========================= Stopping Evaluation =========================")
#         interrupted = True
    
#         if not interrupted:
#             print("=========================Completed=========================")


#     if not load_checkpoint:
#         x = [i+1 for i in range(n_games)]
#         plt.plot(x, score_history)


#     # obs = env.reset(env)








#     # agent_time_elapsed = 0

#     # step_start = time.perf_counter()
#     # actions = agent.register_reset(obs_dict)
#     # agent_time_elapsed += time.perf_counter()- step_start

#     # episodes_completed = 0
#     # num_steps = 0
#     # interrupted = False
#     # episode_metrics = []
#     # actions_taken = []
#     # rewards_given = []
#     # try:
#     #     observations = obs_dict['observation']
#     #     #print(f"Initial observations {observations}")
#     #     #print(f"Initial observations shape is {len(observations[0])}")
#     #     while True:
#     #         observations_, reward, done, _ = env.step(actions)
#     #         #print(f"New observations shape is {len(observations_[0])}")

#     #         if done:
#     #             episodes_completed += 1
#     #             metrics_t = env.evaluate()
#     #             metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
#     #             if np.any(np.isnan(metrics_t)):
#     #                 raise ValueError("Episode metrics are nan, please contant organizers")
#     #             episode_metrics.append(metrics)
#     #             print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

#     #             obs_dict = env_reset(env)

#     #             step_start = time.perf_counter()
#     #             actions = agent.register_reset(obs_dict)
#     #             agent_time_elapsed += time.perf_counter()- step_start
#     #         else:
#     #             step_start = time.perf_counter()
#     #             actions = agent.compute_action(observations, reward, observations_, done)
#     #             agent_time_elapsed += time.perf_counter()- step_start
#     #         # Keep track of actions and rewards
#     #         actions = np.array(actions)#.reshape(1, -1)
#     #         print(actions)
#     #         #print(actions.shape)
#     #         actions_taken.append(actions)
#     #         rewards_given.append(reward)
#     #         observations = observations_

            
# #             num_steps += 1
# #             if num_steps % 1000 == 0:
# #                 print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

# #             if episodes_completed >= Constants.episodes:
# #                 # actions in a 48 hour period
# #                 ## TODO: complete below
# #                 # print("*******************")
# #                 # img_mat = np.array(actions_taken).reshape(-1, 5)
# #                 # print(img_mat)
# #                 # print(img_mat.shape)
# #                 # print("*******************")
# #                 # plt.figure(figsize=(6,8),dpi=80)
# #                 # plt.imshow(np.array(actions_taken),cmap='jet',interpolation='nearest',vmin=-1,vmax=1)
# #                 # plt.colorbar()
# #                 # plt.savefig('sac_test.png')
# #                 break
# #     except KeyboardInterrupt:
# #         print("========================= Stopping Evaluation =========================")
# #         interrupted = True
    
# #     if not interrupted:
# #         print("=========================Completed=========================")

# #     if len(episode_metrics) > 0:
# #         print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
# #         print("Average Emmision Cost:", np.mean([e['emmision_cost'] for e in episode_metrics]))
# #     print(f"Total time taken by agent: {agent_time_elapsed}s")
    

# if __name__ == '__main__':
#     evaluate()
