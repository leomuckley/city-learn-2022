import sys,os
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
sys.path.append('./agents/SAC/')

"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.orderenforcingwrapper import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv

class Constants:
    episodes = 1
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation": observations }
    return obs_dict

def evaluate():
    print("Starting local evaluation")
    
    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent()

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter()- step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    actions_taken = []
    rewards_given = []
    try:
        observations = obs_dict['observation']
        #print(f"Initial observations {observations}")
        #print(f"Initial observations shape is {len(observations[0])}")
        while True:
            observations_, reward, done, _ = env.step(actions)
            #print(f"New observations shape is {len(observations_[0])}")

            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contant organizers")
                episode_metrics.append(metrics)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

                obs_dict = env_reset(env)

                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter()- step_start
            else:
                step_start = time.perf_counter()
                actions = agent.compute_action(observations, reward, observations_, done)
                agent_time_elapsed += time.perf_counter()- step_start
            # Keep track of actions and rewards
            actions = np.array(actions)#.reshape(1, -1)
            print(actions)
            #print(actions.shape)
            actions_taken.append(actions)
            rewards_given.append(reward)
            observations = observations_

            
            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= Constants.episodes:
                # actions in a 48 hour period
                ## TODO: complete below
                # print("*******************")
                # img_mat = np.array(actions_taken).reshape(-1, 5)
                # print(img_mat)
                # print(img_mat.shape)
                # print("*******************")
                # plt.figure(figsize=(6,8),dpi=80)
                # plt.imshow(np.array(actions_taken),cmap='jet',interpolation='nearest',vmin=-1,vmax=1)
                # plt.colorbar()
                # plt.savefig('sac_test.png')
                break
    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True
    
    if not interrupted:
        print("=========================Completed=========================")

    if len(episode_metrics) > 0:
        print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
        print("Average Emmision Cost:", np.mean([e['emmision_cost'] for e in episode_metrics]))
    print(f"Total time taken by agent: {agent_time_elapsed}s")
    

if __name__ == '__main__':
    evaluate()
