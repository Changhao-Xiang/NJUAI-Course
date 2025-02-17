import os
import time

import gym
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from algo import *
from arguments import get_args
from env import Make_Env
from gym_minigrid.wrappers import *
from PIL import Image

t = str(time.time())


def plot(record, n, m, h, start_planning):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record["steps"], record["mean"], color="blue", label="reward")
    ax.fill_between(record["steps"], record["min"], record["max"], color="blue", alpha=0.2)
    ax.set_xlabel("number of steps")
    ax.set_ylabel("Average score per episode")

    exp_name = "exp2_improve2"
    os.makedirs(f"./{exp_name}/start_planning{start_planning}", exist_ok=True)
    fig.savefig(f"./{exp_name}/start_planning{start_planning}/n={n}_m={m}_h={h}.png")
    plt.close("all")


def main():
    # load hyper parameters
    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {"steps": [0], "max": [0], "mean": [0], "min": [0]}

    # environment initial
    envs = Make_Env(env_mode=2)
    action_shape = envs.action_shape
    observation_shape = envs.state_shape
    print(action_shape, observation_shape)

    epsilon = 0.2
    alpha = 0.2
    gamma = 0.99
    n = 3
    m = 3
    start_planning = 1
    h = 5

    # agent initialization
    # you should finish your agent with QAgent
    # e.g. agent = myQAgent()
    agent = myQAgent(action_shape, envs.grid_size, lr=alpha, discount_factor=gamma)

    # model initialization
    # dynamics_model = DynaModel(8, 8, policy=agent)
    dynamics_model = NetworkModel(8, 8, policy=agent)

    # start to train your agent
    for i in range(num_updates):
        # an example of interacting with the environment
        obs = envs.reset()
        obs = obs.astype(int)
        for step in range(args.num_steps):
            # Sample actions with epsilon greedy policy

            if np.random.rand() < epsilon:
                action = envs.action_sample()
            else:
                action = agent.select_action(obs)

            # interact with the environment
            obs_next, reward, done, info = envs.step(action)
            obs_next = obs_next.astype(int)
            # add your Q-learning algorithm
            agent.update(obs, action, reward, obs_next)
            dynamics_model.store_transition(obs, action, reward, obs_next)
            obs = obs_next
            if done:
                obs = envs.reset()

        if i > start_planning:
            for _ in range(n):
                s = dynamics_model.sample_state()
                # buf_tuple = dynamics_model.buffer[idx]
                if isinstance(s, tuple):
                    s = s[0]
                for _ in range(h):
                    if np.random.rand() < epsilon:
                        a = envs.action_sample()
                    else:
                        a = agent.select_action(s)
                    s_ = dynamics_model.predict(s, a)
                    r = envs.R(s, a, s_)
                    done = envs.D(s, a, s_)
                    # add your Q-learning algorithm
                    agent.update(s, a, r, s_)
                    s = s_
                    if done:
                        break

        for _ in range(m):
            dynamics_model.train_transition(32)

        if (i + 1) % (args.log_interval) == 0:
            total_num_steps = (i + 1) * args.num_steps
            obs = envs.reset()
            obs = obs.astype(int)
            reward_episode_set = []
            reward_episode = 0.0
            for step in range(args.test_steps):
                action = agent.select_action(obs)
                obs_next, reward, done, info = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0.0
                    obs = envs.reset()

            end = time.time()
            print(
                "TIME {} Updates {}, num timesteps {}, FPS {} \n avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                    i,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    np.mean(reward_episode_set),
                    np.min(reward_episode_set),
                    np.max(reward_episode_set),
                )
            )
            record["steps"].append(total_num_steps)
            record["mean"].append(np.mean(reward_episode_set))
            record["max"].append(np.max(reward_episode_set))
            record["min"].append(np.min(reward_episode_set))
            plot(record, n, m, h, start_planning)


if __name__ == "__main__":
    main()
