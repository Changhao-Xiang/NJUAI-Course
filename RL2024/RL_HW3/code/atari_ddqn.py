import argparse
import os
import random

import numpy as np
import torch
from buffer import RolloutStorage
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from config import Config
from core.util import get_class_attr_val
from model import CnnDQN, DuelingCnnDQN
from tester import Tester
from torch.optim import Adam
from trainer import Trainer

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class CnnDDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = RolloutStorage(config)

        if "dueling" in self.config.model.lower():
            self.model = DuelingCnnDQN(self.config.state_shape, self.config.action_dim)
            self.target_model = DuelingCnnDQN(self.config.state_shape, self.config.action_dim)
        else:
            self.model = CnnDQN(self.config.state_shape, self.config.action_dim)
            self.target_model = CnnDQN(self.config.state_shape, self.config.action_dim)

        self.target_model.load_state_dict(self.model.state_dict())
        # self.model_optim = torch.optim.RMSprop(
        #     self.model.parameters(),
        #     lr=self.config.learning_rate,
        #     eps=1e-5,
        #     weight_decay=0.95,
        #     momentum=0,
        #     centered=True,
        # )
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.99))

        if self.config.use_cuda:
            self.cuda()

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float) / 255.0
            if self.config.use_cuda:
                state = state.to(self.config.device)
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr):
        s0, s1, a, r, done = self.buffer.sample(self.config.batch_size)
        if self.config.use_cuda:
            s0 = s0.float().to(self.config.device) / 255.0
            s1 = s1.float().to(self.config.device) / 255.0
            a = a.to(self.config.device)
            r = r.to(self.config.device)
            done = done.to(self.config.device)

        # How to calculate Q(s,a) for all actions
        # q_values is a vector with size (batch_size, action_shape, 1)
        # each dimension i represents Q(s0,a_i)
        q_values = self.model(s0).cuda()

        # How to calculate argmax_a Q(s,a)
        actions = q_values.max(1)[1]

        # Tips: function torch.gather may be helpful
        # You need to design how to calculate the loss
        q_values = q_values.gather(1, a).squeeze(1)
        next_q_values = self.target_model(s1).max(1)[0].detach()
        expected_q_values = r.reshape(-1) + self.config.gamma * next_q_values * (1 - done).reshape(-1)
        loss = torch.nn.MSELoss()(q_values, expected_q_values)

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    def cuda(self):
        self.model.to(self.config.device)
        self.target_model.to(self.config.device)

    def load_weights(self, model_path):
        model = torch.load(model_path)
        if "model" in model:
            self.model.load_state_dict(model["model"])
        else:
            self.model.load_state_dict(model)

    def save_model(self, output, name=""):
        torch.save(self.model.state_dict(), "%s/model_%s.pkl" % (output, name))

    def save_config(self, output):
        with open(output + "/config.txt", "w") as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + "/checkpoint_model"
        os.makedirs(checkpath, exist_ok=True)
        torch.save(
            {"frames": fr, "model": self.model.state_dict()}, "%s/checkpoint_fr_%d.tar" % (checkpath, fr)
        )

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint["frames"]
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["model"])
        return fr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--train", dest="train", action="store_true", help="train model")
    parser.add_argument("--env", default="PongNoFrameskip-v4", type=str, help="gym environment")
    parser.add_argument("--test", dest="test", action="store_true", help="test model")
    parser.add_argument("--retrain", dest="retrain", action="store_true", help="retrain model")
    parser.add_argument("--model_path", type=str, help="if test or retrain, import the model")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--cuda_id", type=str, default="0", help="if test or retrain, import the model")

    parser.add_argument(
        "--model", type=str, default="DQN", choices=["DQN", "Dueling_DQN"], help="DQN or Dueling_DQN"
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    config = Config()
    config.env = args.env
    config.model = args.model
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.05
    config.eps_decay = 30000
    config.frames = 2000000
    config.use_cuda = args.cuda
    config.learning_rate = 1e-4
    config.init_buff = 10000
    config.max_buff = 100000
    config.learning_interval = 4
    config.update_tar_interval = 1000
    config.batch_size = 64
    config.gif_interval = 20000
    config.print_interval = 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 500000
    config.win_reward = 18
    config.win_break = True
    config.device = torch.device("cuda:" + args.cuda_id if args.cuda else "cpu")
    # handle the atari env
    env = make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape
    print(config.action_dim, config.state_shape)
    agent = CnnDDQNAgent(config)

    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print("please add the model path:", "--model_path xxxx")
            exit(0)
        tester = Tester(agent, env, args.model_path)
        tester.test(visualize=False)

    elif args.retrain:
        if args.model_path is None:
            print("please add the model path:", "--model_path xxxx")
            exit(0)

        fr = agent.load_checkpoint(args.model_path)
        trainer = Trainer(agent, env, config)
        trainer.train(fr)
