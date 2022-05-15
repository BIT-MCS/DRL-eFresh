import argparse
import gym
import torch

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch.multiprocessing as mp

from utils import spatial_att_github


class Policy(nn.Module):
    def __init__(self, agent):
        super(Policy, self).__init__()
        self.agent = agent

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, observation):
        """Sample action from agents output distribution over actions.
        """
        # Unsqueeze to give a batch size of 1.
        state = torch.from_numpy(observation).float().unsqueeze(0)
        action_scores, _ = self.agent(state)
        action_probs = F.softmax(action_scores, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return action.item()


def finish_episode(optimizer, policy, config):
    """Updates model using REINFORCE.
    """
    eps = np.finfo(np.float32).eps.item()
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + config.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def train(rank, agent, config):
    env = gym.make("Seaquest-v0")
    torch.manual_seed(config.seed+rank)
    env.seed(config.seed+rank)

    policy = Policy(agent=agent)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    running_reward = 10.0

    # NOTE: I am using a different update mechanism as of now (REINFORCE vs. A3C).
    for i_episode in range(config.num_episodes):
        observation = env.reset()
        # resets hidden states, otherwise the comp. graph history spans episodes
        # and relies on freed buffers.
        agent.reset() # NOTE: This may be problematic across processes.
        ep_reward = 0

        # Stash model in case of crash.
        if i_episode % config.save_model_interval == 0 and i_episode > 0:
            torch.save(agent.state_dict(), f"./models/agent-{i_episode}-{rank}.pt")

        for t in range(config.max_steps):
            action = policy(observation)
            reward = 0.0
            for _ in range(config.num_repeat_action):
                if config.render:
                    env.render()
                observation, _reward, done, _ = env.step(action)
                reward += _reward
                if done:
                    break
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
                finish_episode(optimizer, policy, config)
                if i_episode % config.log_interval == 0:
                    print(
                        f"Episode {i_episode}-{rank}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}"
                    )
                if running_reward > config.reward_threshold:
                    print(
                        f"Solved! Running reward is now {running_reward} and "
                        f"the last episode runs to {t} time steps!"
                    )
                break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=1_000)
    parser.add_argument("--num_repeat_action", type=int, default=4)
    parser.add_argument("--reward_threshold", type=int, default=1_000)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--seed", type=int, default=543, metavar="N", help="random seed (default: 543)"
    )
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="interval between training status logs (default: 10)",
    )
    parser.add_argument(
        "--save-model-interval",
        type=int,
        default=250,
        help="interval between saving models.",
    )
    config = parser.parse_args()

    env = gym.make("Seaquest-v0")
    num_actions = env.action_space.n
    agent = spatial_att_github.Agent(num_actions=num_actions)
    # Hogwild Reinforce.
    agent.share_memory()
    mp.spawn(fn=train, args=(agent, config), nprocs=config.num_agents)
    torch.save(agent.state_dict(), f"./models/agent-final.pt")
    
