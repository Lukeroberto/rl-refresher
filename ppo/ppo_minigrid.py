import os
import random
import time

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import minigrid

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# Run PPO on carpole, acrobot and mountain car with 3 seeds/ 9 workers
# Inspired + reference code: https://github.com/vwxyzjn/gym_minigrid/blob/master/ppo.py

args = {
    # Experiment options
    "exp_name": os.path.basename(__file__)[:-len(".py")],
    "seed": 1,
    "torch_deterministic": True,
    "cuda": False,
    "capture_video": False,

    # Algorithm params
    "env_id": "MiniGrid-FourRooms-v0",
    "total_timesteps": 1_000_000,
    "learning_rate": 3e-4,
    "num_envs":  4,
    "num_steps": 128, # length of policy rollout
    "anneal_lr": True,
    "gamma": 0.99, # horizon of roughly 100 steps
    "gae_lambda": 0.95, # lambda for general advantage estimation
    "num_minibatches": 4,
    "update_epochs": 4,
    "norm_adv": True, # normalize advantage estimation
    "clip_coef": 0.2, # surrogate clipping coeff
    "clip_vloss": True, # whether or not to use clipped loss for value function
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None, # max norm for kl div threshold

    # runtime params
    "batch_size": 0,
    "minibatch_size": 0,
    "num_iterations": 0
}

def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b

class FlattenObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        o, _ = self.env.reset()
        obs = np.append(o['image'].flatten()/255., [one_hot(o['direction'], 4)])
        self.observation_space = gym.spaces.Box(0, 1, obs.shape)

    def reset(self, **kwargs):
        o, info = super().reset(**kwargs)
        obs = np.append(o['image'].flatten()/255., [one_hot(o['direction'], 4)])
        return obs, info

    def step(self, action):
        o, reward, terminations, truncations, infos = super().step(action)
        obs = np.append(o['image'].flatten()/255., [one_hot(o['direction'], 4)])
        return obs, reward, terminations, truncations, infos

def make_env(env_id, idx, capture_video, run_name):
    def temp():
        env = gym.make(env_id)
        env = FlattenObsWrapper(env)
        env = RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = RecordVideo(env, f"videos/{run_name}")
        return env
    return temp

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # fcn_ in torch are in-place modifications
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
    
    def value(self, x):
        return self.critic(x)

    def action_and_value(self, x, action=None):
        # Optionally sample an action from actor, return the associated log prob and entropy of action distribution
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":
    args["batch_size"] = int(args["num_envs"] * args["num_steps"])
    args["minibatch_size"] = int(args["batch_size"] // args["num_minibatches"])
    args["num_iterations"] = args["total_timesteps"] // args["batch_size"]

    run_name = f"{args['env_id']}_{args['exp_name']}__{args['seed']}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in args.items()])),
    )

    # Seeds
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.deterministic = args["torch_deterministic"]

    device = torch.device("cuda" if torch.cuda.is_available() and args["cuda"] else "cpu")

    # Setup envs
    env_list = [make_env(args["env_id"], i, args["capture_video"], run_name) for i in range(args["num_envs"])]
    envs = gym.vector.SyncVectorEnv(env_list)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action spaces supported"


    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args["learning_rate"], eps=1e-5)

    n_steps = args["num_steps"]
    n_envs = args["num_envs"]

    obs =      torch.zeros((n_steps, n_envs) + envs.single_observation_space.shape).to(device)
    actions =  torch.zeros((n_steps, n_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((n_steps, n_envs)).to(device)
    rewards =  torch.zeros((n_steps, n_envs)).to(device) 
    dones =    torch.zeros((n_steps, n_envs)).to(device) 
    values =   torch.zeros((n_steps, n_envs)).to(device) 

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args["seed"])
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(n_envs).to(device)

    for iteration in range(1, args["num_iterations"] + 1):
        if args["anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / args["num_iterations"]
            optimizer.param_groups[0]["lr"] = frac * args["learning_rate"]

        for step in range(0, n_steps):
            global_step += n_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Compute advantages aand returns
        with torch.no_grad():
            next_value = agent.value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args["gamma"] * args["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1) 
        b_returns = returns.reshape(-1) 
        b_values = values.reshape(-1) 

        b_inds = np.arange(args["batch_size"])
        clipfracs = []

        # Optimize policy and value network
        for epoch in range(args["update_epochs"]):
            np.random.shuffle(b_inds)

            for start in range(0, args["batch_size"], args["minibatch_size"]):
                end = start + args["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args["clip_coef"], 1 + args["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args["clip_coef"],
                        args["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args["ent_coef"] * entropy_loss + v_loss * args["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args["max_grad_norm"])
                optimizer.step()

            if args["target_kl"] is not None and approx_kl > args["target_kl"]:
                break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        print("SPS: ", int(global_step / (time.time() - start_time)))

        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()












