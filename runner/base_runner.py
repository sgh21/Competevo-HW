import gymnasium as gym
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import competevo
import gym_compete
import os
import shutil

class BaseRunner:
    def __init__(self, cfg, logger, dtype, device, num_threads=1, training=True, ckpt_dir=None, ckpt=0) -> None:
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training
        self.env_name = cfg.env_name
        # dirs
        self.run_dir = logger.run_dir
        self.model_dir = logger.model_dir
        self.log_dir = logger.log_dir
        self.tb_dir = logger.tb_dir

        self.t_start = time.time()

        self.noise_rate = 1.0

        self.setup_env(self.env_name)
        self.setup_writer()
        self.setup_learner()

        has_checkpoint = not self._checkpoint_is_zero(ckpt)
        if has_checkpoint or not training:
            self.load_checkpoint(ckpt_dir, ckpt)
            if training and has_checkpoint and ckpt_dir is not None and not getattr(cfg, "resume_run_dir", None):
                self.copy_initial_checkpoint(ckpt_dir, ckpt)

    def _checkpoint_is_zero(self, ckpt):
        if isinstance(ckpt, (list, tuple)):
            return all(self._checkpoint_is_zero(item) for item in ckpt)
        return ckpt is None or ckpt == 0 or ckpt == "0"

    def _checkpoint_name(self, ckpt):
        if isinstance(ckpt, int):
            return 'epoch_%04d.p' % ckpt
        assert isinstance(ckpt, str)
        return ckpt if ckpt.endswith('.p') else ckpt + '.p'

    def _find_checkpoint_file(self, ckpt_dir, idx, ckpt):
        base_dir = ckpt_dir[idx] if isinstance(ckpt_dir, (list, tuple)) else ckpt_dir
        ckpt_name = self._checkpoint_name(ckpt)
        candidates = [
            os.path.join(base_dir, 'agent_%d' % idx, ckpt_name),
            os.path.join(base_dir, ckpt_name),
        ]
        for cp_path in candidates:
            if os.path.exists(cp_path):
                return cp_path
        return candidates[0]

    def copy_initial_checkpoint(self, ckpt_dir, ckpt):
        def copy_checkpoint(source_file, model_dir, new_filename='epoch_0000.p'):
            os.makedirs(model_dir, exist_ok=True)
            target_file = os.path.join(model_dir, new_filename)
            try:
                shutil.copyfile(source_file, target_file)
                self.logger.info(f"Checkpoint {source_file} copied successfully to {target_file}")
            except FileNotFoundError:
                self.logger.critical(f"Source file {source_file} not found.")
            except Exception as e:
                self.logger.critical(f"Error copying checkpoint: {e}")

        if isinstance(ckpt, (list, tuple)):
            for idx, ckpt_i in enumerate(ckpt):
                source_file = self._find_checkpoint_file(ckpt_dir, idx, ckpt_i)
                copy_checkpoint(source_file, os.path.join(self.model_dir, 'agent_%d' % idx))
        else:
            source_file = self._find_checkpoint_file(ckpt_dir, 0, ckpt)
            copy_checkpoint(source_file, self.model_dir)

    def setup_env(self, env_name):
        env_kwargs = dict(getattr(self.cfg, "env_kwargs", dict()))
        if self.training:
            # self.env = gym.make(env_name, cfg=self.cfg, render_mode="human")
            self.env = gym.make(env_name, cfg=self.cfg, **env_kwargs)
        else:
            render_kwargs = {
                "render_mode": "human",
                "width": getattr(self.cfg, "render_width", 1280),
                "height": getattr(self.cfg, "render_height", 960),
                "default_camera_config": getattr(
                    self.cfg,
                    "default_camera_config",
                    {
                        "distance": 9.0,
                        "azimuth": 90.0,
                        "elevation": -45.0,
                    },
                ),
            }
            render_kwargs.update(env_kwargs)
            self.env = gym.make(env_name, cfg=self.cfg, **render_kwargs)
            # self.env = gym.make(env_name, cfg=self.cfg)

    def setup_writer(self):
        self.writer = SummaryWriter(log_dir=self.tb_dir) if self.training else None

    def setup_learner(self):
        raise NotImplementedError

    def optimize(self, epoch):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
    
    def load_checkpoint(self, ckpt_dir, ckpt):
        raise NotImplementedError
    
    def seed_worker(self, pid):
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            # if hasattr(self.env, 'np_random'):
            #     self.env.np_random.seed(self.env.np_random.randint(5000) * pid)
