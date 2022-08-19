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