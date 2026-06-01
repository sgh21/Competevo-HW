# AGENTS.md

## 项目理解

本仓库当前 `mujoco` 分支是 CompetEvo 的 MuJoCo/Gymnasium 实现，用于 IJCAI-2024 论文
“CompetEvo: Towards Morphological Evolution from Competition” 的形态与策略共同进化实验。

这个分支已经明显不同于 `master` 的 IsaacGym/rl-games/Hydra 栈。当前代码主要围绕：

- MuJoCo XML 资产与动态形态构造；
- Gymnasium 环境注册与多智能体环境；
- 自定义 PPO/采样/runner 训练循环；
- fixed morphology、developed morphology、evolved morphology 的 run-to-goal 与 robo-sumo 任务。

不要把 `master` 分支中的 IsaacGym 目录、Hydra 配置、`rl_games` 训练入口当成当前分支的事实来源。

## 运行入口与常用命令

本地开发默认使用 conda 环境 `EAI`，不使用 Docker。

常用命令：

```bash
conda activate EAI
python train.py --cfg config/run-to-goal-ants-v0.yaml
python display.py --cfg config/robo-sumo-devants-v0.yaml --ckpt_dir runs/robo-sumo-devants-v0/models
python test_robot.py
```

依赖安装：

```bash
conda activate EAI
python -m pip install -r requirements.txt
```

如果没有激活环境，可以使用：

```bash
conda run -n EAI python train.py --cfg config/run-to-goal-ants-v0.yaml
conda run -n EAI python display.py --cfg config/robo-sumo-devants-v0.yaml --ckpt_dir runs/robo-sumo-devants-v0/models
```

## 重要配置文件

- `config/config.py`：轻量 YAML 配置加载器，负责把任务、优化器、runner、形态进化参数展开成属性。
- `config/*.yaml`：实验配置。`env_name` 对应 `competevo/__init__.py` 或 `gym_compete/__init__.py` 中注册的 Gymnasium id。
- `requirements.txt`：本地 conda 环境安装清单。当前环境是 Python 3.11，PyTorch 使用 CUDA 12 系列轮子。
- `docker/requirements.txt`、`docker/dockerfile`：历史 Docker 环境线索。不要直接照搬其中的 Python 3.8/CUDA 11.3/PyTorch 1.12 组合到本地 `EAI` 环境。
- `runs/robo-sumo-devants-v0/`：仓库内附带的示例配置和预训练 checkpoint，可用于 display smoke test。

## 核心模块与算法

- `train.py`：训练入口。解析 `--cfg`，创建 `Config`、`Logger`，按 `runner_type` 选择 runner。
- `display.py`：加载 checkpoint 并以 `render_mode="human"` 运行环境展示。
- `runner/base_runner.py`：runner 基类，负责创建 Gymnasium 环境、TensorBoard writer、加载 checkpoint。
- `runner/multi_agent_runner.py`：固定形态多智能体 runner。
- `runner/multi_evo_agent_runner.py`：形态进化/发育型 agent runner。
- `runner/selfplay_agent_runner.py`：self-play runner。
- `custom/learners/`：采样与 PPO 更新逻辑。
- `custom/models/`：普通 actor/critic、development actor/critic、Transform2Act actor/critic、GNN/JSMLP。
- `lib/rl/core/`：分布、policy、critic、advantage、trajectory batch、running norm 等 RL 基础组件。
- `lib/utils/`：数学、torch、memory、MuJoCo、统计日志等工具。
- `competevo/__init__.py`：注册 CompetEvo 自定义 Gymnasium 环境，例如 `robo-sumo-devants-v0`、`run-to-goal-evoants-v0`。
- `competevo/evo_envs/`：CompetEvo 形态进化环境、agent 定义、XML 资产和 XML 生成工具。
- `gym_compete/`：OpenAI multiagent-competition 风格环境的本地改写，包括 humanoid/ant/bug/spider 等基础 agent。

已知代码注意点：

- `lib/rl/agents/*` 与 `lib/rl/envs/visual/humanoid_vis.py` 中仍有旧的 `khrylib.*` import 残留；当前主入口不依赖这些模块，修改时不要误判为当前 runner 的必经路径。
- `gymnasium==0.28.1` 与 `mujoco==2.3.5` 是 Docker 清单中的关键组合，升级时要重点验证环境 reset、step、render。
- `Config.out_dir` 当前写死为 `/root/ws/competevo/tmp`，本地运行如需长期保存日志，优先通过 logger 或配置方式调整，避免把路径散落到代码里。

## 代码修改原则

1. 修改前先理解相关目录、调用链、测试和现有风格。
2. 只做与当前任务直接相关的最小修改，不重写无关模块。
3. 不改变核心算法逻辑，除非任务明确要求且有小规模验证。
4. 生成文件优先写入形成可追溯易管理文件夹，不要散落在根目录。
5. 遇到不确定模块作用时，在文档或汇报中标注“待确认”，不要猜测。
6. 修改环境注册、XML 生成、观测维度、动作维度、reward 或 done 条件后，至少跑一次对应 `gym.make(...).reset()` smoke test。
7. 修改训练循环或模型结构后，优先使用小 batch/短 epoch 配置做快速验证，不直接启动长训练。
8. 不提交 `tmp/`、新 checkpoint、大日志或本地环境产物，除非任务明确要求保留。
9. 代码实现后进行代码检查，清除中间测试的的临时代码残留，仅保留必要修改。


## 每次任务完成后的汇报格式

任务结束时用中文汇报：

1. 修改了什么
2. 为什么这样修改
3. 修改了哪些文件
4. 执行了哪些命令
5. 测试或验证结果
6. 当前仍不确定的信息
7. 下一步建议
