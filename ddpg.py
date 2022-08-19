from rewards.user_reward import UserReward
import numpy as np
import math
import time, random, typing, cProfile, traceback
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import pandas as pd

from citylearn.citylearn import CityLearnEnv
from comm_net import CommNet
from critc import SingleCritic
from utils import Queue, MinMaxNormalizer

class Constants:
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
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict

env = CityLearnEnv(schema=Constants.schema_path)
obs_dict = env_reset(env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG:
    MEMORY_SIZE = 10000
    BATCH_SIZE = 128
    GAMMA = 0.95
    LR = 3e-4
    TAU = 0.001

    memory = Queue()

    def to(self,device):
        self.device = device
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)


    def __init__(self, obs_dict):

            
        N = len(obs_dict['building_info'])
        obs_len = len(obs_dict['observation_space'][0]['high'])

        # Initalize actor networks
        self.actor = CommNet(
        agent_number=N,
        input_size=obs_len
        )

        self.actor_target = copy.deepcopy(self.actor)

        # Initialize critic networks
        self.critic = SingleCritic(
            input_size=obs_len*N,
            action_size=N
        )

        self.critic_target = copy.deepcopy(self.critic)

        self.normalizer = MinMaxNormalizer(obs_dict=obs_dict)

        self.c_criterion = nn.MSELoss()
        self.a_optimize = optim.Adam(self.actor.parameters(),lr=self.LR)
        self.c_optimize = optim.Adam(self.critic.parameters(),lr=self.LR)

        self.to("cpu")
        
    def compute_action(self, obs, exploration=True, exploration_factor = 0.3):
        obs = self.normalizer.transform(obs)
        action = self.actor(torch.tensor(obs,device=self.device).float()).detach().cpu().numpy()
        # Adding some exploration noise
        if exploration:
            action = action + np.random.normal(scale=exploration_factor,size = action.shape)
            action = np.clip(action,a_min=-1.0,a_max=1.0)
        return action

    def add_memory(self, s, a, r, ns):
        s = self.normalizer.transform(s)
        ns = self.normalizer.transform(ns)
        self.memory.enqueue([s,a,r,ns])
        if len(self.memory) > self.MEMORY_SIZE:
            self.memory.dequeue()

    def clear_memory(self):
        self.memory.a = []
        self.memory.b = []

    # Conduct an update step to the policy
    def update(self):
        torch.set_grad_enabled(True)

        N = self.BATCH_SIZE
        if len(self.memory) < 1: # Watch before learn
            return 
        # Get a minibatch of experiences
        # mb = random.sample(self.memory, min(len(self.memory),N)) # This is slow with a large memory size
        mb = []
        for _ in range(min(len(self.memory),N)):
            mb.append(self.memory[random.randint(0,len(self.memory)-1)])

        s = torch.tensor(np.array([x[0] for x in mb]),device=self.device).float()
        a = torch.tensor(np.array([x[1] for x in mb]),device=self.device).float()
        r = torch.tensor(np.array([x[2] for x in mb]),device=self.device).float()
        ns = torch.tensor(np.array([x[3] for x in mb]),device=self.device).float()

        # Critic update
        self.c_optimize.zero_grad()
        nsa = self.actor_target.forward(ns,batch=True)
        y_t = torch.add(torch.unsqueeze(r,1), self.GAMMA * self.critic_target(ns,nsa))
        y_c = self.critic(s,a) 
        c_loss = self.c_criterion(y_c,y_t)
        critic_loss = c_loss.item()
        c_loss.backward()
        self.c_optimize.step()

        # Actor update
        self.a_optimize.zero_grad()
        a_loss = -self.critic(s,self.actor.forward(s,batch=True)).mean() # Maximize gradient direction increasing objective function
        a_loss.backward()
        self.a_optimize.step()

        # Target networks
        for ct_p, c_p in zip(self.critic_target.parameters(), self.critic.parameters()):
            ct_p.data = ct_p.data * (1.0-self.TAU) + c_p.data * self.TAU

        for at_p, a_p in zip(self.actor_target.parameters(), self.actor.parameters()):
            at_p.data = at_p.data * (1.0-self.TAU) + a_p.data * self.TAU

        torch.set_grad_enabled(False)

        return critic_loss





def train_ddpg(
    agent : DDPG,
    env,
    num_iterations = 50000,
    debug = True,
    evaluation = False,
    exploration_decay = 0.001
    ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dict = env.reset()
    start = time.time()
    print(f'Starting training at {start}')
    rewards = []
    episode_metrics = []
    episodes_completed = 0
    loss = 0

    observations = env.observations
    actions = agent.compute_action(observations, exploration=False)

    try:
        steps = 0
        reward = 0
        while steps < num_iterations+1:
            steps += 1
            
            decay = max(0.05,0,5*math.exp(-exploration_decay*steps))
            prev_observations = observations
            observations, reward, done, _ = env.step(actions)
            # std, mean = (1.5508584091038358, -1.5304271926841968) # Precomputed
            # reward = (UserReward(agent_count=len(observations),observation=observations).calculate()[0] - mean) / std

            # TODO: Integrate with Neptune
            if True:
                rewards.append(reward)
                if debug:
                    if steps % 480 == 1 and steps > 1:
                        print('Time {} Episode {} Step {}: Reward {} Actions {} Loss {}'.format(time.time()-start,episodes_completed, steps,np.array(rewards[-24:]).mean(),np.array(actions).T, loss))
                reward = 0
                
            if done:
                reward = 0
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contant organizers")
                episode_metrics.append(metrics)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

                env_reset(env)
                observations = env.observations
                actions = agent.compute_action(observations, exploration=False)
            else:
                agent.add_memory(
                    s=prev_observations,
                    a=actions,
                    r=reward,
                    ns=observations
                )
                actions = agent.compute_action(observations,exploration_factor=decay)
                if not evaluation:
                    loss = agent.update()
                
    except Exception as e:
        if debug:
            traceback.print_exc()
        else:
            print(e)
    
    print(f"Training finished at {time.time()-start}")
    return rewards, episode_metrics



if __name__ == "__main__":
    ddpg_agent = DDPG(obs_dict)
    DURATION = 24*365*3
    rewards, episode_metrics = train_ddpg(agent=ddpg_agent, env = env, num_iterations=DURATION)