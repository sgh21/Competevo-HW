import gymnasium as gym
from config.config import Config
import argparse
import numpy as np
import torch
import gc

import logging
from logger.logger import Logger
from utils.tools import *

import time
import sys, os
sys.path.append(".")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from runner.multi_evo_agent_runner import MultiEvoAgentRunner
from runner.multi_agent_runner import MultiAgentRunner
from runner.selfplay_agent_runner import SPAgentRunner


def parse_agent_ids(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ("", "none", "fixed", "no", "false"):
        return []
    if value in ("all", "both", "true"):
        return [0, 1]
    return [int(item.strip()) for item in value.split(",") if item.strip() != ""]


def set_cfg_attr(cfg, name, value):
    if value is None:
        return
    setattr(cfg, name, value)
    cfg.cfg[name] = value


def apply_cli_overrides(cfg, args):
    if args.game_mode is not None:
        game_mode = args.game_mode.replace("-", "_")
        set_cfg_attr(cfg, "game_mode", game_mode)
        if game_mode == "selfplay":
            set_cfg_attr(cfg, "runner_type", "selfplay-agent-runner")
        elif game_mode in ("two_player", "twoplayer"):
            set_cfg_attr(cfg, "runner_type", "multi-evo-agent-runner")
        else:
            raise ValueError("Unsupported game mode: %s" % args.game_mode)

    if args.morph_optim_agents is not None:
        morph_agents = parse_agent_ids(args.morph_optim_agents)
        set_cfg_attr(cfg, "morph_optim_agents", morph_agents)

    if args.reward_mode is not None:
        cfg.reward_specs = dict(cfg.reward_specs)
        cfg.reward_specs["mode"] = args.reward_mode
        cfg.cfg["reward_specs"] = cfg.reward_specs
        if args.reward_mode == "run_to_goal_warmup":
            set_cfg_attr(cfg, "use_parse_reward", False)
            set_cfg_attr(cfg, "use_exploration_curriculum", False)

    simple_overrides = [
        "run_label",
        "max_epoch_num",
        "min_batch_size",
        "mini_batch_size",
        "eval_batch_size",
        "eval_num_threads",
        "num_optim_epoch",
        "save_model_interval",
        "termination_epoch",
        "delta",
        "use_opponent_sample",
        "use_exploration_curriculum",
        "use_parse_reward",
    ]
    for name in simple_overrides:
        set_cfg_attr(cfg, name, getattr(args, name))


def main():
    # ----------------------------------------------------------------------------#
    # Load config options from terminal and predefined yaml file
    # ----------------------------------------------------------------------------#
    parser = argparse.ArgumentParser(description="User's arguments from terminal.")
    parser.add_argument("--cfg", 
                        dest="cfg_file", 
                        help="Config file", 
                        required=True, 
                        type=str)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default='0')
    parser.add_argument('--game_mode', type=str, default=None,
                        help='selfplay or two_player; overrides runner_type')
    parser.add_argument('--morph_optim_agents', type=str, default=None,
                        help='none, all, 0, 1, or comma-separated ids such as 0,1')
    parser.add_argument('--reward_mode', type=str, default=None,
                        help='sumo or run_to_goal_warmup')
    parser.add_argument('--run_label', type=str, default=None,
                        help='human-readable prefix for the run directory')
    parser.add_argument('--max_epoch_num', type=int, default=None)
    parser.add_argument('--min_batch_size', type=int, default=None)
    parser.add_argument('--mini_batch_size', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--eval_num_threads', type=int, default=None)
    parser.add_argument('--num_optim_epoch', type=int, default=None)
    parser.add_argument('--save_model_interval', type=int, default=None)
    parser.add_argument('--termination_epoch', type=int, default=None)
    parser.add_argument('--delta', type=float, default=None)
    parser.add_argument('--use_opponent_sample', type=str2bool, default=None)
    parser.add_argument('--use_exploration_curriculum', type=str2bool, default=None)
    parser.add_argument('--use_parse_reward', type=str2bool, default=None)
    args = parser.parse_args()
    # Load config file
    cfg = Config(args.cfg_file)
    apply_cli_overrides(cfg, args)

    # ----------------------------------------------------------------------------#
    # Define logger and create dirs
    # ----------------------------------------------------------------------------#
    logger = Logger(name='current', cfg=cfg)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # set output
    logger.set_output_handler()
    logger.print_system_info()
    # only training generates log file
    logger.critical("The current environment is {}.".format(cfg.env_name))
    logger.info("Running directory: {}".format(logger.run_dir))
    logger.info('Type of current running: Training')
    logger.set_file_handler()
    # Save the config file
    cfg.save_config(logger.run_dir)

    # ----------------------------------------------------------------------------#
    # Set torch and random seed
    # ----------------------------------------------------------------------------#
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) \
        if args.use_cuda and torch.cuda.is_available() else torch.device('cpu')
    # torch.cuda.is_available() is natively False on mac m1
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # ----------------------------------------------------------------------------#
    # Training
    # ----------------------------------------------------------------------------#
    # runner definition
    # runner = MultiEvoAgentRunner(cfg, logger, dtype, device, 
    #                              num_threads=args.num_threads, training=True)
    
    ckpt = int(args.ckpt) if args.ckpt.isnumeric() else args.ckpt

    if cfg.runner_type == "multi-agent-runner":
        ckpt = [ckpt] * 2
        runner = MultiAgentRunner(cfg, logger, dtype, device, 
                                    num_threads=args.num_threads, training=True, ckpt_dir=args.ckpt_dir, ckpt=ckpt)
    elif cfg.runner_type == "selfplay-agent-runner":
        runner = SPAgentRunner(cfg, logger, dtype, device, 
                                    num_threads=args.num_threads, training=True, ckpt_dir=args.ckpt_dir, ckpt=ckpt)
    elif cfg.runner_type == "multi-evo-agent-runner":
        ckpt = [ckpt] * 2
        runner = MultiEvoAgentRunner(cfg, logger, dtype, device,
                                     num_threads=args.num_threads, training=True, ckpt_dir=args.ckpt_dir, ckpt=ckpt)
    
    # main loop
    for epoch in range(0, cfg.max_epoch_num):
        runner.optimize(epoch)
        runner.save_checkpoint(epoch)

        """clean up gpu memory"""
        gc.collect()
        torch.cuda.empty_cache()

    runner.logger.info('training done!')

if __name__ == "__main__":
    main()
