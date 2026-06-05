import argparse
import csv
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from itertools import combinations

# Headless smoke tests need a MuJoCo GL backend before mujoco is imported.
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

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


REWARD_KEYS = [
    "reward_parse",
    "reward_dense",
    "win_reward",
    "lose_penalty",
    "ctrl_reward",
    "alive_reward",
    "move_to_opp_reward",
    "push_opp_reward",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_sampler(cfg, dtype, device, agent):
    flag = getattr(agent, "flag", None)
    if flag == "dev":
        return DevSampler(cfg, dtype, device, agent)
    if flag == "evo":
        return EvoSampler(cfg, dtype, device, agent)
    return Sampler(cfg, dtype, device, agent)


def to_torch_state(states, device):
    result = []
    for state in states:
        if isinstance(state, list):
            result.append([torch.tensor(x).to(device) for x in state])
        else:
            result.append(torch.tensor(state).to(device))
    return result


def latest_dir(pattern):
    dirs = [path for path in glob.glob(pattern) if os.path.isdir(path)]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def require_file(path, desc):
    if not os.path.exists(path):
        raise FileNotFoundError("%s not found: %s" % (desc, path))
    return path


def agent_entry(agent_id, display_name, training_line, training_mode, source_role,
                run_dir, checkpoint_file, morph_enabled):
    return {
        "agent_id": agent_id,
        "display_name": display_name,
        "training_line": training_line,
        "training_mode": training_mode,
        "source_role": source_role,
        "run_dir": run_dir,
        "checkpoint": "best",
        "checkpoint_file": checkpoint_file,
        "morph_enabled": bool(morph_enabled),
        "fixed_morph_scale": 0.0,
        "train_cfg": os.path.join(run_dir, "config.yml"),
    }


def discover_formal_best_agents(root="tmp/robo-sumo-devants-v0"):
    patterns = {
        "sp_fixed": [
            os.path.join(root, "formal-parallel-*-sp-fixed-*"),
            os.path.join(root, "formal-sp-fixed-*"),
        ],
        "tp_fixed": [
            os.path.join(root, "formal-parallel-*-tp-fixed-*"),
            os.path.join(root, "formal-tp-fixed-*"),
        ],
        "sp_morph": [
            os.path.join(root, "formal-parallel-*-sp-morph-*"),
            os.path.join(root, "formal-sp-morph-*"),
        ],
        "tp_morph": [
            os.path.join(root, "formal-parallel-*-tp-morph-*"),
            os.path.join(root, "formal-tp-morph-*"),
        ],
        "tp_mixed": [
            os.path.join(root, "formal-parallel-*-tp-mixed-a0morph-*"),
            os.path.join(root, "formal-tp-mixed-a0morph-*"),
        ],
    }
    run_dirs = {}
    for key, pats in patterns.items():
        for pattern in pats:
            run_dir = latest_dir(pattern)
            if run_dir is not None:
                run_dirs[key] = run_dir
                break
        if key not in run_dirs:
            raise FileNotFoundError("No run directory found for %s" % key)

    sp_fixed = run_dirs["sp_fixed"]
    tp_fixed = run_dirs["tp_fixed"]
    sp_morph = run_dirs["sp_morph"]
    tp_morph = run_dirs["tp_morph"]
    tp_mixed = run_dirs["tp_mixed"]

    agents = [
        agent_entry(
            "sp_fixed",
            "Self-play fixed morphology",
            "sp-fixed",
            "selfplay",
            "selfplay",
            sp_fixed,
            require_file(os.path.join(sp_fixed, "models", "best.p"), "sp_fixed best checkpoint"),
            False,
        ),
        agent_entry(
            "tp_fixed_a0",
            "Two-player fixed morphology agent0",
            "tp-fixed",
            "two_player",
            "agent_0",
            tp_fixed,
            require_file(os.path.join(tp_fixed, "models", "agent_0", "best.p"), "tp_fixed_a0 best checkpoint"),
            False,
        ),
        agent_entry(
            "tp_fixed_a1",
            "Two-player fixed morphology agent1",
            "tp-fixed",
            "two_player",
            "agent_1",
            tp_fixed,
            require_file(os.path.join(tp_fixed, "models", "agent_1", "best.p"), "tp_fixed_a1 best checkpoint"),
            False,
        ),
        agent_entry(
            "sp_morph",
            "Self-play optimized morphology",
            "sp-morph",
            "selfplay",
            "selfplay",
            sp_morph,
            require_file(os.path.join(sp_morph, "models", "best.p"), "sp_morph best checkpoint"),
            True,
        ),
        agent_entry(
            "tp_morph_a0",
            "Two-player optimized morphology agent0",
            "tp-morph",
            "two_player",
            "agent_0",
            tp_morph,
            require_file(os.path.join(tp_morph, "models", "agent_0", "best.p"), "tp_morph_a0 best checkpoint"),
            True,
        ),
        agent_entry(
            "tp_morph_a1",
            "Two-player optimized morphology agent1",
            "tp-morph",
            "two_player",
            "agent_1",
            tp_morph,
            require_file(os.path.join(tp_morph, "models", "agent_1", "best.p"), "tp_morph_a1 best checkpoint"),
            True,
        ),
        agent_entry(
            "tp_mixed_a0_morph",
            "Mixed two-player agent0 morphology optimized",
            "tp-mixed-a0morph",
            "two_player",
            "agent_0",
            tp_mixed,
            require_file(os.path.join(tp_mixed, "models", "agent_0", "best.p"), "tp_mixed_a0_morph best checkpoint"),
            True,
        ),
        agent_entry(
            "tp_mixed_a1_fixed",
            "Mixed two-player agent1 fixed morphology",
            "tp-mixed-a0morph",
            "two_player",
            "agent_1",
            tp_mixed,
            require_file(os.path.join(tp_mixed, "models", "agent_1", "best.p"), "tp_mixed_a1_fixed best checkpoint"),
            False,
        ),
    ]
    return agents


def build_eval_cfg(base_cfg, slot_agents):
    cfg = Config(base_cfg)
    morph_agents = []
    for slot, agent in enumerate(slot_agents):
        if agent["morph_enabled"]:
            morph_agents.append(slot)

    cfg.runner_type = "multi-evo-agent-runner"
    cfg.game_mode = "two_player"
    cfg.reward_specs = dict(cfg.reward_specs)
    cfg.reward_specs["mode"] = "sumo"
    cfg.use_parse_reward = True
    cfg.use_exploration_curriculum = False
    cfg.use_opponent_sample = False
    cfg.morph_optim_agents = morph_agents
    cfg.fixed_morph_scale = 0.0
    cfg.render_width = 1280
    cfg.render_height = 960
    cfg.default_camera_config = {
        "distance": 9.0,
        "azimuth": 90.0,
        "elevation": -45.0,
        "lookat": np.array([0.0, 0.0, 0.5]),
    }

    cfg.cfg["runner_type"] = cfg.runner_type
    cfg.cfg["game_mode"] = cfg.game_mode
    cfg.cfg["reward_specs"] = cfg.reward_specs
    cfg.cfg["use_parse_reward"] = cfg.use_parse_reward
    cfg.cfg["use_exploration_curriculum"] = cfg.use_exploration_curriculum
    cfg.cfg["use_opponent_sample"] = cfg.use_opponent_sample
    cfg.cfg["morph_optim_agents"] = cfg.morph_optim_agents
    cfg.cfg["fixed_morph_scale"] = cfg.fixed_morph_scale
    return cfg


def make_env(cfg, render_mode=None):
    kwargs = {}
    if render_mode:
        kwargs.update({
            "render_mode": render_mode,
            "width": getattr(cfg, "render_width", 1280),
            "height": getattr(cfg, "render_height", 960),
            "default_camera_config": getattr(cfg, "default_camera_config", None),
        })
    return gym.make(cfg.env_name, cfg=cfg, **kwargs)


def load_samplers(cfg, env, agents, dtype, device):
    samplers = {}
    checkpoint_files = {}
    for i, env_agent in env.agents.items():
        sampler = build_sampler(cfg, dtype, device, env_agent)
        cp_path = agents[i]["checkpoint_file"]
        with open(cp_path, "rb") as f:
            sampler.load_ckpt(pickle.load(f))
        samplers[i] = sampler
        checkpoint_files[str(i)] = cp_path
    return samplers, checkpoint_files


def select_actions(states, samplers, mean_action, device):
    state_var = to_torch_state(states, device)
    actions = []
    with torch.no_grad():
        for i, sampler in samplers.items():
            if getattr(sampler, "flag", None) in ("dev", "evo"):
                action = sampler.policy_net.select_action([state_var[i]], mean_action)
            else:
                action = sampler.policy_net.select_action(state_var[i], mean_action)
            actions.append(action.detach().cpu().squeeze().numpy().astype(np.float64))
    return actions


def normalize_next_states(next_states, samplers, device):
    result = list(next_states)
    for i, sampler in samplers.items():
        if sampler.running_state is not None and not isinstance(result[i], list):
            result[i] = sampler.running_state(result[i])
    return tuple(result)


def agent_side(env, idx):
    x = float(env.agents[idx].get_qpos()[0])
    if x < 0:
        return "left"
    if x > 0:
        return "right"
    return "center"


def reward_components(info):
    return {key: float(info.get(key, 0.0)) for key in REWARD_KEYS}


def safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def safe_std(values):
    return float(np.std(values)) if values else 0.0


def determine_winner(infos):
    winners = [i for i, info in enumerate(infos) if "winner" in info]
    if len(winners) == 1:
        return winners[0]
    return None


def capture_frame(env):
    if not hasattr(env, "env_scene"):
        return None
    frame = env.env_scene.render()
    if frame is None:
        return None
    return np.asarray(frame)


def evaluate_fixture(fixture, base_cfg, episodes, seed, mean_action=True,
                     record_video=False, video_path=None, video_episodes=1,
                     video_fps=30, video_stride=2):
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device("cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = build_eval_cfg(base_cfg, fixture["slot_agents"])
    env = make_env(cfg, render_mode="rgb_array" if record_video else None)
    samplers, checkpoint_files = load_samplers(cfg, env, fixture["slot_agents"], dtype, device)

    episode_records = []
    video_meta = None
    writer = None
    frames_written = 0
    if record_video:
        import imageio
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = imageio.get_writer(video_path, fps=video_fps)

    try:
        for ep in range(episodes):
            states, _ = env.reset()
            states = normalize_next_states(states, samplers, device)
            episode_reward = [0.0 for _ in samplers]
            initial_sides = None
            last_infos = None
            truncated_flag = False
            winner = None
            length = 0

            for t in range(10000):
                actions = select_actions(states, samplers, mean_action, device)
                next_states, env_rewards, terminateds, truncated, infos = env.step(actions)
                last_infos = infos
                truncated_flag = bool(truncated)

                if initial_sides is None and getattr(env, "stage", None) == "execution":
                    initial_sides = [agent_side(env, i) for i in range(len(samplers))]

                for i, reward in enumerate(env_rewards):
                    episode_reward[i] += float(reward)

                if record_video and ep < video_episodes and getattr(env, "stage", None) == "execution":
                    if t % max(video_stride, 1) == 0:
                        frame = capture_frame(env)
                        if frame is not None:
                            writer.append_data(frame)
                            frames_written += 1

                states = normalize_next_states(next_states, samplers, device)
                length = t + 1
                if terminateds[0] or truncated:
                    winner = determine_winner(infos)
                    break

            if initial_sides is None:
                initial_sides = [agent_side(env, i) for i in range(len(samplers))]
            components = [reward_components(info) for info in last_infos] if last_infos else [{}, {}]
            if winner is None:
                result = "draw"
            elif winner == 0:
                result = "slot0_win"
            else:
                result = "slot1_win"
            episode_records.append({
                "episode_index": ep,
                "result": result,
                "winner_slot": winner,
                "rewards": episode_reward,
                "reward_margin_slot0": float(episode_reward[0] - episode_reward[1]),
                "length": length,
                "truncated": truncated_flag,
                "initial_sides": initial_sides,
                "reward_components": components,
            })
    finally:
        if writer is not None:
            writer.close()
            video_meta = {
                "path": video_path,
                "episodes": min(video_episodes, episodes),
                "fps": video_fps,
                "stride": video_stride,
                "frames_written": frames_written,
            }
        env.close()

    summary = summarize_fixture_episodes(episode_records)
    return {
        "fixture_id": fixture["fixture_id"],
        "pair_id": fixture["pair_id"],
        "slot0_agent_id": fixture["slot_agents"][0]["agent_id"],
        "slot1_agent_id": fixture["slot_agents"][1]["agent_id"],
        "seed": seed,
        "episodes_requested": episodes,
        "morph_optim_agents": cfg.morph_optim_agents,
        "checkpoint_files": checkpoint_files,
        "summary": summary,
        "episodes": episode_records,
        "video": video_meta,
    }


def summarize_fixture_episodes(episodes):
    wins = [0, 0]
    draws = 0
    rewards = [[], []]
    lengths = []
    margins = []
    side_counts = {"slot0": defaultdict(int), "slot1": defaultdict(int)}
    side_wins = {"slot0": defaultdict(int), "slot1": defaultdict(int)}

    for ep in episodes:
        for i in range(2):
            rewards[i].append(float(ep["rewards"][i]))
        lengths.append(int(ep["length"]))
        margins.append(float(ep["reward_margin_slot0"]))
        if ep["winner_slot"] is None:
            draws += 1
        else:
            wins[ep["winner_slot"]] += 1
        for slot in range(2):
            key = "slot%d" % slot
            side = ep["initial_sides"][slot]
            side_counts[key][side] += 1
            if ep["winner_slot"] == slot:
                side_wins[key][side] += 1

    n = len(episodes)
    return {
        "episodes": n,
        "wins": wins,
        "draws": draws,
        "win_rate": [wins[0] / n if n else 0.0, wins[1] / n if n else 0.0],
        "draw_rate": draws / n if n else 0.0,
        "avg_episode_reward": [safe_mean(rewards[0]), safe_mean(rewards[1])],
        "std_episode_reward": [safe_std(rewards[0]), safe_std(rewards[1])],
        "avg_reward_margin_slot0": safe_mean(margins),
        "avg_episode_length": safe_mean(lengths),
        "side_counts": {
            slot: dict(counts) for slot, counts in side_counts.items()
        },
        "side_win_counts": {
            slot: dict(counts) for slot, counts in side_wins.items()
        },
    }


def build_fixtures(agents, smoke=False, smoke_pairs=1):
    fixtures = []
    pairs = list(combinations(range(len(agents)), 2))
    if smoke:
        pairs = pairs[:max(1, smoke_pairs)]
    for pair_index, (a_idx, b_idx) in enumerate(pairs):
        a = agents[a_idx]
        b = agents[b_idx]
        pair_id = "%s__vs__%s" % (a["agent_id"], b["agent_id"])
        fixtures.append({
            "fixture_id": "%s__leg0" % pair_id,
            "pair_id": pair_id,
            "slot_agents": [a, b],
        })
        fixtures.append({
            "fixture_id": "%s__leg1" % pair_id,
            "pair_id": pair_id,
            "slot_agents": [b, a],
        })
    return fixtures


def empty_stats(agent):
    return {
        "agent_id": agent["agent_id"],
        "display_name": agent["display_name"],
        "points": 0,
        "episodes": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "reward_sum": 0.0,
        "reward_sq_sum": 0.0,
        "reward_margin_sum": 0.0,
        "episode_length_sum": 0.0,
        "slot_counts": {"agent0": 0, "agent1": 0},
        "slot_wins": {"agent0": 0, "agent1": 0},
        "side_counts": defaultdict(int),
        "side_wins": defaultdict(int),
    }


def aggregate_results(agents, fixture_results):
    stats = {agent["agent_id"]: empty_stats(agent) for agent in agents}
    matrix = {
        a["agent_id"]: {
            b["agent_id"]: {
                "episodes": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "reward_margin_sum": 0.0,
            }
            for b in agents if b["agent_id"] != a["agent_id"]
        }
        for a in agents
    }

    for fixture in fixture_results:
        slot_ids = [fixture["slot0_agent_id"], fixture["slot1_agent_id"]]
        for ep in fixture["episodes"]:
            rewards = ep["rewards"]
            length = ep["length"]
            winner = ep["winner_slot"]
            for slot in range(2):
                agent_id = slot_ids[slot]
                opp_id = slot_ids[1 - slot]
                st = stats[agent_id]
                st["episodes"] += 1
                st["reward_sum"] += float(rewards[slot])
                st["reward_sq_sum"] += float(rewards[slot]) ** 2
                st["reward_margin_sum"] += float(rewards[slot] - rewards[1 - slot])
                st["episode_length_sum"] += float(length)
                slot_key = "agent%d" % slot
                st["slot_counts"][slot_key] += 1
                side = ep["initial_sides"][slot]
                st["side_counts"][side] += 1

                cell = matrix[agent_id][opp_id]
                cell["episodes"] += 1
                cell["reward_margin_sum"] += float(rewards[slot] - rewards[1 - slot])

                if winner is None:
                    st["draws"] += 1
                    st["points"] += 1
                    cell["draws"] += 1
                elif winner == slot:
                    st["wins"] += 1
                    st["points"] += 3
                    st["slot_wins"][slot_key] += 1
                    st["side_wins"][side] += 1
                    cell["wins"] += 1
                else:
                    st["losses"] += 1
                    cell["losses"] += 1

    leaderboard = []
    for agent in agents:
        st = stats[agent["agent_id"]]
        n = st["episodes"]
        reward_mean = st["reward_sum"] / n if n else 0.0
        reward_var = st["reward_sq_sum"] / n - reward_mean ** 2 if n else 0.0
        side_counts = dict(st["side_counts"])
        side_wins = dict(st["side_wins"])
        entry = {
            "agent_id": st["agent_id"],
            "display_name": st["display_name"],
            "points": st["points"],
            "points_per_episode": st["points"] / n if n else 0.0,
            "episodes": n,
            "wins": st["wins"],
            "draws": st["draws"],
            "losses": st["losses"],
            "win_rate": st["wins"] / n if n else 0.0,
            "draw_rate": st["draws"] / n if n else 0.0,
            "loss_rate": st["losses"] / n if n else 0.0,
            "avg_reward": reward_mean,
            "std_reward": float(max(reward_var, 0.0) ** 0.5),
            "avg_reward_margin": st["reward_margin_sum"] / n if n else 0.0,
            "avg_episode_length": st["episode_length_sum"] / n if n else 0.0,
            "agent0_win_rate": (
                st["slot_wins"]["agent0"] / st["slot_counts"]["agent0"]
                if st["slot_counts"]["agent0"] else 0.0
            ),
            "agent1_win_rate": (
                st["slot_wins"]["agent1"] / st["slot_counts"]["agent1"]
                if st["slot_counts"]["agent1"] else 0.0
            ),
            "left_win_rate": (
                side_wins.get("left", 0) / side_counts.get("left", 0)
                if side_counts.get("left", 0) else 0.0
            ),
            "right_win_rate": (
                side_wins.get("right", 0) / side_counts.get("right", 0)
                if side_counts.get("right", 0) else 0.0
            ),
            "side_counts": side_counts,
            "slot_counts": st["slot_counts"],
        }
        leaderboard.append(entry)

    leaderboard.sort(
        key=lambda x: (x["points"], x["win_rate"], x["avg_reward_margin"]),
        reverse=True,
    )
    for rank, entry in enumerate(leaderboard, 1):
        entry["rank"] = rank

    for row in matrix.values():
        for cell in row.values():
            n = cell["episodes"]
            cell["win_rate"] = cell["wins"] / n if n else 0.0
            cell["draw_rate"] = cell["draws"] / n if n else 0.0
            cell["loss_rate"] = cell["losses"] / n if n else 0.0
            cell["avg_reward_margin"] = cell["reward_margin_sum"] / n if n else 0.0

    return leaderboard, matrix


def write_leaderboard_csv(path, leaderboard):
    fieldnames = [
        "rank",
        "agent_id",
        "display_name",
        "points",
        "points_per_episode",
        "episodes",
        "wins",
        "draws",
        "losses",
        "win_rate",
        "draw_rate",
        "loss_rate",
        "avg_reward",
        "std_reward",
        "avg_reward_margin",
        "avg_episode_length",
        "agent0_win_rate",
        "agent1_win_rate",
        "left_win_rate",
        "right_win_rate",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in leaderboard:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_fixtures_csv(path, fixture_results):
    fieldnames = [
        "fixture_id",
        "pair_id",
        "slot0_agent_id",
        "slot1_agent_id",
        "seed",
        "episodes",
        "slot0_wins",
        "slot1_wins",
        "draws",
        "slot0_win_rate",
        "slot1_win_rate",
        "draw_rate",
        "slot0_avg_reward",
        "slot1_avg_reward",
        "avg_reward_margin_slot0",
        "avg_episode_length",
        "video_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fixture in fixture_results:
            s = fixture["summary"]
            row = {
                "fixture_id": fixture["fixture_id"],
                "pair_id": fixture["pair_id"],
                "slot0_agent_id": fixture["slot0_agent_id"],
                "slot1_agent_id": fixture["slot1_agent_id"],
                "seed": fixture["seed"],
                "episodes": s["episodes"],
                "slot0_wins": s["wins"][0],
                "slot1_wins": s["wins"][1],
                "draws": s["draws"],
                "slot0_win_rate": s["win_rate"][0],
                "slot1_win_rate": s["win_rate"][1],
                "draw_rate": s["draw_rate"],
                "slot0_avg_reward": s["avg_episode_reward"][0],
                "slot1_avg_reward": s["avg_episode_reward"][1],
                "avg_reward_margin_slot0": s["avg_reward_margin_slot0"],
                "avg_episode_length": s["avg_episode_length"],
                "video_path": fixture["video"]["path"] if fixture.get("video") else "",
            }
            writer.writerow(row)


def video_filename(fixture, seed):
    slot0 = fixture["slot_agents"][0]["agent_id"]
    slot1 = fixture["slot_agents"][1]["agent_id"]
    return "%s__slot0_%s__slot1_%s__seed%d.mp4" % (
        fixture["pair_id"],
        slot0,
        slot1,
        seed,
    )


def add_video_context(video_meta, result):
    if not video_meta:
        return None
    enriched = dict(video_meta)
    enriched.update({
        "fixture_id": result["fixture_id"],
        "pair_id": result["pair_id"],
        "slot0_agent_id": result["slot0_agent_id"],
        "slot1_agent_id": result["slot1_agent_id"],
        "seed": result["seed"],
    })
    return enriched


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, defaultdict):
        return dict(obj)
    raise TypeError("Object of type %s is not JSON serializable" % type(obj).__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="formal_best", choices=["formal_best"])
    parser.add_argument("--base_cfg", default="config/repro/unified-devant-training.yaml")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_pairs", type=int, default=1)
    parser.add_argument("--video_pair", nargs=2, default=None)
    parser.add_argument("--video_all_pairs", action="store_true")
    parser.add_argument("--video_episodes", type=int, default=1)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_stride", type=int, default=2)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--formal_root", default="tmp/robo-sumo-devants-v0")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "videos"), exist_ok=True)

    agents = discover_formal_best_agents(args.formal_root)
    agent_by_id = {agent["agent_id"]: agent for agent in agents}
    fixtures = build_fixtures(agents, smoke=args.smoke, smoke_pairs=args.smoke_pairs)

    video_pair = tuple(args.video_pair) if args.video_pair else None
    if video_pair and args.video_all_pairs:
        raise ValueError("Use either --video_pair or --video_all_pairs, not both.")
    if video_pair:
        for agent_id in video_pair:
            if agent_id not in agent_by_id:
                raise ValueError("Unknown video_pair agent id: %s" % agent_id)

    fixture_results = []
    video_records = []
    video_exhibitions = []
    recorded_video_pair_ids = set()
    mean_action = not args.stochastic

    for index, fixture in enumerate(fixtures):
        slot_ids = [fixture["slot_agents"][0]["agent_id"], fixture["slot_agents"][1]["agent_id"]]
        fixture_seed = args.seed + index
        should_record = False
        video_path = None
        if args.video_all_pairs and fixture["pair_id"] not in recorded_video_pair_ids:
            should_record = True
            recorded_video_pair_ids.add(fixture["pair_id"])
            video_path = os.path.join(
                args.out_dir,
                "videos",
                video_filename(fixture, fixture_seed),
            )
        elif video_pair and tuple(slot_ids) == video_pair:
            should_record = True
            video_path = os.path.join(
                args.out_dir,
                "videos",
                video_filename(fixture, fixture_seed),
            )
        result = evaluate_fixture(
            fixture,
            args.base_cfg,
            args.episodes,
            fixture_seed,
            mean_action=mean_action,
            record_video=should_record,
            video_path=video_path,
            video_episodes=args.video_episodes,
            video_fps=args.video_fps,
            video_stride=args.video_stride,
        )
        fixture_results.append(result)
        if result.get("video"):
            video_records.append(add_video_context(result["video"], result))
        print(json.dumps({
            "fixture_id": result["fixture_id"],
            "slot0": result["slot0_agent_id"],
            "slot1": result["slot1_agent_id"],
            "summary": result["summary"],
            "video": result["video"],
        }, indent=2, default=json_default))

    if video_pair and not video_records:
        extra_fixture = {
            "fixture_id": "%s__vs__%s__video_only" % video_pair,
            "pair_id": "%s__vs__%s" % video_pair,
            "slot_agents": [agent_by_id[video_pair[0]], agent_by_id[video_pair[1]]],
        }
        video_path = os.path.join(
            args.out_dir,
            "videos",
            "%s__%s__video_only_seed%d.mp4" % (video_pair[0], video_pair[1], args.seed + 100000),
        )
        result = evaluate_fixture(
            extra_fixture,
            args.base_cfg,
            args.video_episodes,
            args.seed + 100000,
            mean_action=mean_action,
            record_video=True,
            video_path=video_path,
            video_episodes=args.video_episodes,
            video_fps=args.video_fps,
            video_stride=args.video_stride,
        )
        result["video_only"] = True
        video_exhibitions.append(result)
        video_records.append(add_video_context(result["video"], result))

    leaderboard, match_matrix = aggregate_results(agents, fixture_results)

    result = {
        "settings": {
            "preset": args.preset,
            "base_cfg": args.base_cfg,
            "out_dir": args.out_dir,
            "episodes": args.episodes,
            "seed": args.seed,
            "smoke": args.smoke,
            "smoke_pairs": args.smoke_pairs,
            "mean_action": mean_action,
            "video_pair": list(video_pair) if video_pair else None,
            "video_all_pairs": args.video_all_pairs,
            "video_episodes": args.video_episodes,
            "video_fps": args.video_fps,
            "video_stride": args.video_stride,
        },
        "agents": agents,
        "fixtures": fixture_results,
        "video_exhibitions": video_exhibitions,
        "leaderboard": leaderboard,
        "match_matrix": match_matrix,
        "videos": video_records,
    }

    results_path = os.path.join(args.out_dir, "league_results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=json_default)

    write_leaderboard_csv(os.path.join(args.out_dir, "leaderboard.csv"), leaderboard)
    write_fixtures_csv(os.path.join(args.out_dir, "fixtures.csv"), fixture_results)
    print("Wrote %s" % results_path)


if __name__ == "__main__":
    main()
