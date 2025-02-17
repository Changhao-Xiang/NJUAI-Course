from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import SGD, Adam
from torchvision.models import resnet
from tqdm import tqdm


class DaggerAgent:
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
    def __init__(self, necessary_parameters=None):
        super(DaggerAgent, self).__init__()
        # init your model
        self.model = None

    # train your model with labeled data
    def update(self, data_batch, label_batch):
        self.model.train(data_batch, label_batch)

    # select actions by your model
    def select_action(self, data_batch):
        label_predict = self.model.predict(data_batch)
        return label_predict


class CNNDaggerAgent(DaggerAgent):
    def __init__(self, observation_shape, action_shape=8):
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (observation_shape[0] // 8) * (observation_shape[1] // 8), action_shape),
        )
        # comstrain actions in MontezumaRevengeNoFrameskip-v0
        self.output2action = [0, 1, 2, 3, 4, 5, 11, 12]

    def update(self, data_batch, label_batch):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        self.model.to(device)
        self.model.train()

        optimizer = SGD(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for data, label in tqdm(zip(data_batch, label_batch), total=len(label_batch)):
            data = torch.tensor(data, dtype=torch.float32).to(device)
            data = data.permute(2, 0, 1).unsqueeze(0)
            label = self.output2action.index(label)
            label = torch.tensor([int(label)], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = self.model(data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

    def select_action(self, ob):
        self.model.eval()
        input_tensor = torch.tensor(ob, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        logits = self.model(input_tensor)
        print(logits)
        output = torch.argmax(logits)
        action = self.output2action[output]

        return action
