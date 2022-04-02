import torch
import torch.nn as nn
import numpy as np
import gym

class WorldModel(gym.Wrapper):
    def __init__(self, env, maxstep = 100):
        super().__init__(env)
        self.model = NN(env.observation_space.shape[0], env.action_space.shape[0])
        self.optim = torch.optim.Adam(self.model.parameters())
        self._max_episode_steps = maxstep

    def train(self):
        buffer = Buffer(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        mse_loss = nn.MSELoss()
        for i in range(5):
            obs = self.env.reset()
            done = False
            while not done:
                act = self.env.action_space.sample()
                n_obs, reward, done, info = self.env.step(act)

                x = np.concatenate((obs, act))
                buffer.append(x,n_obs)

            X,Y = buffer.sample()

            X = torch.from_numpy(X.astype(np.float32)).clone()
            Y = torch.from_numpy(Y.astype(np.float32)).clone()

            pred = self.model(X)
            loss = mse_loss(Y,pred)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            print("loss", loss.item())
    
    def step(self, action):
        self.count += 1
        done = self.count > self._max_episode_steps

        if type(action) is not np.ndarray:
            action = np.ndarray(action)
        
        x = np.concatenate((self.obs, action))

        x = torch.from_numpy(x.astype(np.float32)).clone()

        n_obs = self.model(x)
        n_obs = n_obs.to('cpu').detach().numpy().copy()
        self.obs = n_obs
        return n_obs, 0, done, {} # done の判定ができないのが良くないかも
    
    def reset(self):
        self.obs = self.env.reset() #あんまよくない。余裕があれば修正
        self.count = 0
        return self.obs

class Buffer():
    def __init__(self, obs_size, action_size):
        self.x_size = obs_size + action_size
        self.y_size = obs_size

        self.len = 10**4
        self.i = 0
        self.memory = np.zeros((self.len, self.x_size + self.y_size))
        self.max = False
    
    def append(self, x,y):
        self.memory[self.i] = np.concatenate((x,y))
        self.i += 1
        if self.i >= self.len:
            self.max = True
            self.i = 0

    def sample(self, num=255):
        ma = self.len-1 if self.max else self.i
        indexes = np.random.randint(0, high=ma, size=(num))
        xy = self.memory[indexes]
        x,y = xy[:,:self.x_size], xy[:,self.x_size:]
        return x, y

class NN(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(obs_space+act_space, 250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.ReLU(),
            nn.Linear(250, 64),
            nn.ReLU(),
            nn.Linear(64, obs_space)
        )
    
    def forward(self, x):
        return self.layer(x)

if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    worldmodel = WorldModel(env)
    worldmodel.train()

    for ep in range(3):
        obs = worldmodel.reset()
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, done, info = worldmodel.step(action)
    