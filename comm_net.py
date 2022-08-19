import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CommNet(nn.Module):
    '''
    Implements CommNet for a single building
    Of the CityLearn challenge
    LSTM version with skip connection for the final layer

    TODO: Try basic version without LSTM / alter skip connections etc
            But might be a better idea to explore more advanced architectures instead
    '''

    def __init__(
                self, 
                agent_number,       # Number of buildings present
                input_size,         # Observation accessible to each building (assuming homogenous)
                hidden_size = 10,   # Hidden vector accessible at each communication step
                comm_size = 4,      # Number of communication channels
                comm_steps = 2      # Number of communication steps
                ):
                
        super(CommNet, self).__init__()

        self.device = 'cpu'
        self.input_size = input_size
        self.comm_size = comm_size
        self.agent_number = agent_number
        self.comm_steps = comm_steps

        # Calculate first hidden layer 
        self._in_mlp = nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size,input_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size,hidden_size)
        )

        # Communication 
        self._lstm = nn.LSTMCell(
            input_size = comm_size,
            hidden_size = hidden_size
        )

        self._comm_mlp = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,comm_size)
        )

        # Output
        # Calculate based on inputs and final memory
        self._out_mlp = nn.Sequential(
            nn.Linear(input_size+hidden_size, input_size+hidden_size),
            nn.LeakyReLU(),
            nn.Linear(input_size+hidden_size, input_size+hidden_size),
            nn.LeakyReLU(),
            nn.Linear(input_size+hidden_size, 1),
            nn.Tanh()
        )


    def forward(self,x : torch.Tensor, batch = False):

        out = None
        if not batch:

            # (Building, Observations)
            
            # Initial hidden states
            hidden_states = self._in_mlp(x)
            cell_states = torch.zeros(hidden_states.shape,device=self.device)

            # Communication
            for t in range(self.comm_steps):
                # Calculate communication vectors
                comm = self._comm_mlp(hidden_states)
                total_comm = torch.sum(comm,0)
                comm = (total_comm - comm) / (self.agent_number-1)
                # Apply LSTM   
                hidden_states, cell_states = self._lstm(comm,(hidden_states,cell_states))
            
            out = self._out_mlp(torch.cat((x,hidden_states),dim=1))
        else:
            # (Batch, Building, Observation)
            out = torch.stack([self.forward(a) for a in x])

        return out

    def to(self,device):
        super().to(device)
        self.device = device

class SingleCritic(nn.Module):

    def __init__(self,
                input_size, 
                action_size = 1,
                hidden_layer_size = 32):
        super(SingleCritic, self).__init__()

        self.input_size = input_size
        self.action_size = action_size

        self._in_mlp = nn.Sequential(
            nn.Linear(input_size + action_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, 1),
        )

    def forward (self, state, action):
        x = torch.cat((torch.flatten(state,start_dim=1),torch.flatten(action,start_dim=1)),dim=1)
        return self._in_mlp(x)

from sklearn.preprocessing import MinMaxScaler

class MinMaxNormalizer:

    def __init__(self, obs_dict):
        observation_space = obs_dict['observation_space'][0]
        low, high = observation_space['low'],observation_space['high']
        
        self.scalar = MinMaxScaler()
        self.scalar.fit([low,high])

    def transform(self, x):
        return self.scalar.transform(x)


# Experience replay needs a memory - this is it!
# Double stack implementation of a queue - https://stackoverflow.com/questions/69192/how-to-implement-a-queue-using-two-stacks
class Queue: 
    a = []
    b = []
    
    def enqueue(self, x):
        self.a.append(x)
    
    def dequeue(self):
        if len(self.b) == 0:
            while len(self.a) > 0:
                self.b.append(self.a.pop())
        if len(self.b):
            return self.b.pop()

    def __len__(self):
        return len(self.a) + len(self.b)

    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError
        if i < len(self.b):
            return self.b[-i-1]
        else:
            return self.a[i-len(self.b)]

