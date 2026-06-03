import argparse
import json
import os
import pickle
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.append(".")

import competevo  # noqa: F401
import gym_compete  # noqa: F401
from config.config import Config
from custom.learners.dev_sampler import DevSampler
from custom.learners.evo_sampler import EvoSampler
from custom.learners.sampler import Sampler


def build_sampler(cfg, dtype, device, agent):
    flag = getattr(agent, "flag", None)
    if flag == "dev":
        return DevSampler(cfg, dtype, device, agent)
    if flag == "evo":
        return EvoSampler(cfg, dtype, device, agent)
    return Sampler(cfg, dtype, device, agent)


def checkpoint_file(ckpt_dir, agent_id, ckpt):
    if ckpt.isdigit():
        name = "epoch_%04d.p" % int(ckpt)
    elif ckpt.endswith(".p"):
        name = ckpt
    else:
        name = ckpt + ".p"
    candidates = [
        os.path.join(ckpt_dir, "agent_%d" % agent_id, name),
        os.path.join(ckpt_dir, name),
    ]
    for cp_path in candidates:
        if os.path.exists(cp_path):
            return cp_path
    return candidates[0]


def selfplay_checkpoint_file(ckpt_dir, ckpt):
    if ckpt.isdigit():
        name = "epoch_%04d.p" % int(ckpt)
    elif ckpt.endswith(".p"):
        name = ckpt
    else:
        name = ckpt + ".p"
    return os.path.join(ckpt_dir, name)


def to_torch_state(states, device):
    result = []
    for state in states:
        if isinstance(state, list):
            result.append([torch.tensor(x).to(device) for x in state])
        else:
            result.append(torch.tensor(state).to(device))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--ckpt", default="best")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--selfplay", action="store_true")
    parser.add_argument("--opponent_ckpt", default=None)
    parser.add_argument("--agent0_ckpt_dir", default=None)
    parser.add_argument("--agent0_ckpt", default=None)
    parser.add_argument("--agent1_ckpt_dir", default=None)
    parser.add_argument("--agent1_ckpt", default=None)
    args = parser.parse_args()

    cfg = Config(args.cfg)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(cfg.env_name, cfg=cfg)
    samplers = {}
    checkpoint_files = {}
    for i, agent in env.agents.items():
        samplers[i] = build_sampler(cfg, dtype, device, agent)
        agent_ckpt_dir = getattr(args, "agent%d_ckpt_dir" % i) or args.ckpt_dir
        agent_ckpt = getattr(args, "agent%d_ckpt" % i) or args.ckpt
        if i > 0 and args.opponent_ckpt is not None:
            agent_ckpt = args.opponent_ckpt

        if (args.selfplay or cfg.runner_type == "selfplay-agent-runner") and \
           getattr(args, "agent%d_ckpt_dir" % i) is None:
            ckpt = args.opponent_ckpt if i > 0 and args.opponent_ckpt is not None else args.ckpt
            cp_path = selfplay_checkpoint_file(agent_ckpt_dir, ckpt)
        else:
            cp_path = checkpoint_file(agent_ckpt_dir, i, agent_ckpt)
        checkpoint_files[str(i)] = cp_path
        with open(cp_path, "rb") as f:
            samplers[i].load_ckpt(pickle.load(f))

    rewards_by_episode = []
    lengths = []
    wins = [0 for _ in samplers]
    draws = 0
    mean_action = not args.stochastic

    for _ in range(args.episodes):
        states, _ = env.reset()
        episode_reward = [0.0 for _ in samplers]
        episode_winner = None
        for t in range(10000):
            state_var = to_torch_state(states, device)
            actions = []
            for i, sampler in samplers.items():
                if sampler.running_state is not None and not isinstance(states[i], list):
                    state_var[i] = torch.tensor(sampler.running_state(states[i])).to(device)
                if getattr(sampler, "flag", None) in ("dev", "evo"):
                    action = sampler.policy_net.select_action([state_var[i]], mean_action)
                else:
                    action = sampler.policy_net.select_action(state_var[i], mean_action)
                actions.append(action.detach().cpu().squeeze().numpy().astype(np.float64))

            states, env_rewards, terminateds, truncated, infos = env.step(actions)
            for i, reward in enumerate(env_rewards):
                episode_reward[i] += float(reward)

            if terminateds[0] or truncated:
                for i, info in enumerate(infos):
                    if "winner" in info:
                        episode_winner = i
                if episode_winner is None:
                    draws += 1
                else:
                    wins[episode_winner] += 1
                lengths.append(t + 1)
                break

        rewards_by_episode.append(episode_reward)

    env.close()
    rewards = np.array(rewards_by_episode, dtype=np.float64)
    result = {
        "config": args.cfg,
        "env_name": cfg.env_name,
        "ckpt_dir": args.ckpt_dir,
        "ckpt": args.ckpt,
        "checkpoint_files": checkpoint_files,
        "episodes": args.episodes,
        "mean_action": mean_action,
        "avg_episode_reward": rewards.mean(axis=0).tolist(),
        "std_episode_reward": rewards.std(axis=0).tolist(),
        "avg_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "wins": wins,
        "draws": draws,
        "win_rate": [w / args.episodes for w in wins],
        "draw_rate": draws / args.episodes,
        "rewards_by_episode": rewards_by_episode,
        "episode_lengths": lengths,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
