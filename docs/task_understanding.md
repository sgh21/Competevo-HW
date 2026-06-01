# CompetEvo 任务理解与讨论记录

本文档用于记录后续讨论形成的任务理解，便于在代码修改前回顾目标、约束、实现入口和待确认点。当前内容是基于仓库扫描与 `ref_doc/paper.pdf` 的初始理解，后续讨论产生的新结论应继续维护在这里。

## 0. 面向讨论的核心理解

CompetEvo 的核心思想不是“给一个固定机器人训练控制器”，而是把“身体长成什么样”和“怎样使用身体对抗”放进同一个强化学习问题里联合优化。

在论文表述中，一个可进化智能体的策略可以理解为两部分：

- morph sub-policy：在每局开始时根据初始形态/当前形态编码生成新的形态参数，得到这一局要使用的身体。
- tactics sub-policy：在这一局对抗过程中，根据自身状态、对手观测和形态编码输出控制动作。

因此，每一局比赛可以拆成两个概念阶段：

1. morph-generation：先生成或调整身体。
2. arena-confrontation：用刚生成的身体进行对抗。

代码里的 `dev_*` 和 `evo_*` 环境正是在模拟这个流程：每个 episode 开始先执行一次或数次形态动作，重建 MuJoCo XML，然后进入执行控制阶段。固定形态 agent 则跳过形态生成，只学习 tactics。

论文中另一个重要概念是 training curriculum，它不是新的网络结构，而是训练顺序与奖励权重安排：

- warm-up：先让 agent 学会基本移动能力，例如朝正确方向快速、低能耗、稳定地移动。
- confrontation training：开始强调胜负对抗。
- rewards annealing：早期 dense reward 权重大，后期 sparse win/loss reward 权重大。

当前代码没有一个自动调度器把 warm-up 和 confrontation 串起来；更接近的使用方式是先跑一个 warm-up 配置得到 checkpoint，再在对抗配置中继承参数继续训练。同时，当前对抗配置中通过 `use_exploration_curriculum` 实现 dense/sparse reward 的平滑退火。

## 0.1 术语和缩写参照表

| 名称 | 全称/来源 | 在本项目中的含义 | 例子或代码位置 |
| --- | --- | --- | --- |
| CompetEvo | Competitive Evolution | 论文提出的方法：在竞争/对抗中联合优化机器人形态和控制策略。 | `ref_doc/paper.pdf` |
| morph | morphology | 机器人身体形态，包括腿长、腿粗、关节能力、gear、骨架拓扑等。 | 论文里的 morph design `M` |
| tactics | fighting tactics / control policy | 使用身体进行对抗的控制策略，即每一步如何驱动 actuator。 | `NormalPolicy`、`DevPolicy`、`Transform2ActPolicy` 的 execution 分支 |
| morph sub-policy | 形态子策略 | 每局开始输出形态参数或骨架动作的策略部分。 | `DevPolicy` 的 scale 分支；`Transform2ActPolicy` 的 skel/attr 分支 |
| tactics sub-policy | 战术/控制子策略 | 对抗过程中输出控制动作的策略部分。 | `control_mlp`、`control_gnn` |
| fixed morph | fixed morphology | 固定形态机器人，只训练控制策略，不改变身体。 | `run-to-goal-ants-v0`、`robo-sumo-ants-v0` |
| `dev_*` | developed morphology agents | 固定拓扑但可调尺寸/属性的发育型机器人。它不会增删身体节点，只缩放已有身体参数。 | `dev_ant`、`dev_bug`、`dev_spider` |
| `evo_*` | evolved morphology agents | 更接近 Transform2Act 的进化型机器人。可以先做骨架结构变换，再做属性参数变换。 | `evo_ant`、`run-to-goal-evoants-v0` |
| `*_fighter` | fighter variant | robo-sumo/对抗场景下的 agent 变体，reward 和观测更偏搏斗。 | `dev_ant_fighter`、`robo_ant_fighter` |
| `devants` | developed ants | 两个 developed ant 的任务配置。 | `robo-sumo-devants-v0` |
| `evoants` | evolved ants | 两个 evolved ant 的任务配置。 | `run-to-goal-evoants-v0` |
| `animals` | mixed animal setting | 混合物种/混合形态任务，可能一个固定形态、一个 dev 形态。 | `robo-sumo-animals-v0` |
| PPO | Proximal Policy Optimization | 当前训练使用的 on-policy 强化学习算法，通过 clipped objective 限制策略更新幅度。 | `custom/learners/*_learner.py` |
| GAE | Generalized Advantage Estimation | 用 value 和 reward 估计 advantage 的方法，用于降低方差。 | `lib/rl/core/common.py` |
| actor / policy | 策略网络 | 输入观测，输出动作分布。 | `NormalPolicy`、`DevPolicy`、`Transform2ActPolicy` |
| critic / value | 价值网络 | 输入状态/观测，输出当前状态价值 `V(s)`。 | `NormalValue`、`DevValue`、`Transform2ActValue` |
| learner | 学习器 | 持有 policy/value/optimizer，负责 PPO 更新。 | `Learner`、`DevLearner`、`EvoLearner` |
| sampler | 采样器 | 只用于 rollout 中加载 policy checkpoint 并采样动作，不做优化。 | `Sampler`、`DevSampler`、`EvoSampler` |
| runner | 训练流程控制器 | 创建环境、组织采样、调用 learner 更新、评估、保存 checkpoint。 | `MultiAgentRunner`、`MultiEvoAgentRunner` |
| episode | 一局/一回合 | 从环境 reset 到 done/truncated 的完整比赛过程。可变形态 episode 开头包含形态生成。 | runner 的采样循环 |
| epoch / generation | 训练迭代/代 | 采样一批数据并更新一次策略的外层迭代。论文常称 generation，代码常称 epoch。 | `for epoch in range(cfg.max_epoch_num)` |
| rollout | 轨迹采样 | 用当前/历史策略在环境中跑若干 step，收集训练数据。 | `sample_worker` |
| checkpoint / ckpt | 模型存档 | 保存 policy/value 参数和归一化状态，用于评估、继续训练和历史对手采样。 | `runs/.../models/agent_*/epoch_*.p` |
| `qpos` | generalized position | MuJoCo 中机器人广义位置，包括根节点位置/姿态和关节位置。 | agent `_get_obs()` |
| `qvel` | generalized velocity | MuJoCo 中机器人广义速度，包括根节点速度和关节速度。 | agent `_get_obs()` |
| `cfrc_ext` | contact external force | MuJoCo 中 body 外部接触力，常作为对抗/碰撞观测的一部分。 | fighter observation |
| actuator | 执行器 | MuJoCo XML 中定义的电机/驱动器，策略输出最终会作用到 actuator。 | XML `actuator/motor` |
| actuator control | 控制量 | 每个 step 发送给 actuator 的连续动作，不一定等同于直接关节角度。 | `env_scene.simulate(actions)` |
| XML | MuJoCo model XML | 描述机器人身体、关节、geom、actuator 和世界的文件/字符串。形态变化会改 XML。 | `assets/*.xml`、`cur_xml_str` |
| `skel` | skeleton | 骨架/拓扑结构，主要指身体节点是否增加或删除。 | `skeleton_transform` |
| `attr` | attribute | 身体属性参数，如 geom size、body offset、actuator gear 等。 | `attribute_transform` |
| design params | design parameters | 形态参数向量，是 morph sub-policy 直接或间接优化的对象。 | `design_params`、`scale_vector` |
| `scale_vector` | 缩放参数向量 | `dev_*` agent 使用的形态缩放向量，通常映射到腿长、腿粗、gear。 | `DevAnt.scale_vector` |
| `robot_param_scale` | 形态动作缩放系数 | 把策略输出的形态动作映射到实际设计参数变化幅度。 | config 中的 `robot_param_scale` |
| `skeleton_transform` | 骨架变换阶段 | `evo_*` episode 开头的阶段，输出 add/remove/none 等离散骨架动作。 | `MultiEvoAgentEnv.step()` |
| `attribute_transform` | 属性变换阶段 | episode 开头的形态参数调整阶段。`dev_*` 只有这一类形态阶段。 | `MultiDevAgentEnv.step()` |
| `execution` | 执行/对抗阶段 | 身体确定后进入物理对抗，每个 step 输出控制动作并获得任务 reward。 | env `_step()` |
| `skel_transform_nsteps` | 骨架变换步数 | `evo_*` 在进入属性变换前执行多少次 skeleton action。 | `config/run-to-goal-evoants-v0.yaml` |
| dense reward | 稠密奖励 | 每一步都有的 shaping reward，用于学移动、稳定、靠近对手、推对手等基本能力。 | `reward_dense` |
| sparse reward | 稀疏奖励 | 胜负奖励，只有达成目标/胜负/平局等关键事件时明显出现。 | `reward_parse` |
| `reward_parse` | parse/sparse reward | 代码里记录胜负类稀疏奖励的字段。名字写作 parse，但语义接近 sparse/task reward。 | env info |
| reward annealing | 奖励退火 | 从 dense reward 主导逐步过渡到 sparse reward 主导。 | `use_exploration_curriculum` |
| `termination_epoch` | 退火结束 epoch | 到这个 epoch 后 `alpha=0`，训练主要依赖 sparse reward。 | config |
| delta-Uniform | 对手历史策略均匀采样 | 从对手历史 checkpoint 池中按 delta 控制的时间窗口随机抽旧策略作为对手。 | `use_opponent_sample`、`delta` |
| `delta` | 历史池起点比例 | 越小历史池越宽，对手更多样；越大越接近当前策略。 | config |
| ego / opponent | 当前训练体/对手 | 某次采样中要更新的 agent 是 ego，另一个加载历史策略的是 opponent。两个 agent 会轮流作为 ego。 | `idx` in runner |
| mean action | 均值动作 | 评估时不用随机采样，而用 Gaussian mean 或 Categorical argmax，表现更稳定。 | `mean_action=True` |
| `use_opponent_sample` | 是否使用历史对手采样 | 开启后训练时 ego 用最新策略，对手从历史 checkpoint 中采样。 | config |
| `use_parse_reward` | 是否使用胜负稀疏奖励 | 关闭后只用 dense reward 或环境 shaping reward。 | config |
| `use_exploration_curriculum` | 是否启用奖励退火 | 开启后使用 dense/sparse 混合奖励。 | config/runner |

## 1. 当前仓库定位

- 当前分支是 CompetEvo 的 MuJoCo/Gymnasium 实现，不应套用 `master` 中 IsaacGym、Hydra、`rl_games` 的训练路径。
- 主训练入口是 `train.py`，配置从 `config/*.yaml` 读取，经 `Config` 展开后按 `runner_type` 选择 runner。
- 环境注册分两类：
  - `gym_compete/__init__.py` 注册固定形态任务，如 `run-to-goal-ants-v0`、`robo-sumo-ants-v0`。
  - `competevo/__init__.py` 注册可变形态任务，如 `run-to-goal-evoants-v0`、`run-to-goal-devants-v0`、`robo-sumo-devants-v0`。

## 2. 论文方法要点

- 论文提出 CompetEvo：在二人对抗任务中联合优化 morphology 和 tactics。
- 每局开始先进行 morphology generation，得到双方形态，再进入 arena confrontation。
- 可进化策略可理解为组合策略 `pi(theta; phi)`：
  - morph sub-policy 生成形态参数；
  - tactics sub-policy 生成对抗过程中的控制动作。
- 训练采用 self-practice / opponent sampling：当前策略与对手历史策略池中按 `delta` 阈值采样的旧策略对战。
- 奖励使用 dense reward 与 sparse reward 的退火混合：
  - `R = kappa * R_dense + (1 - kappa) * R_sparse`
  - `kappa = max((Tt - t) / Tt, 0)`
- PPO 用于每个 epoch 的策略更新。论文描述的超参包括 PPO clipping 0.2、`gamma=0.995`、GAE `tau=0.95`、每 batch 50000 samples、mini-batch 约 2000。

## 3. 训练入口与 runner 链路

- `train.py`
  - 解析 `--cfg`、`--ckpt_dir`、`--ckpt` 等参数。
  - 创建 `Config` 与 `Logger`。
  - 固定 dtype 为 `torch.float64`。
  - 按 `cfg.runner_type` 创建：
    - `MultiAgentRunner`
    - `SPAgentRunner`
    - `MultiEvoAgentRunner`
  - 主循环：`runner.optimize(epoch)` 后 `runner.save_checkpoint(epoch)`。

- `BaseRunner`
  - 创建 gymnasium 环境。
  - training 时无 render；display 时传 `render_mode="human"` 以及窗口参数。
  - 根据 checkpoint 加载 agent_0 / agent_1 模型。

- `MultiAgentRunner`
  - 用于固定形态 agent。
  - 每个 env agent 对应一个 `Learner`。
  - 每 epoch 流程：
    1. 按 `min_batch_size` 采样。
    2. 每个 agent 独立调用 PPO 更新。
    3. 用 mean action 按 `eval_batch_size` 评估。
    4. 记录 reward、win rate、episode length。

- `MultiEvoAgentRunner`
  - 用于含可变形态 agent 的任务。
  - 根据 `agent.flag` 选择 learner：
    - `flag == "evo"` -> `EvoLearner`
    - `flag == "dev"` -> `DevLearner`
    - 否则 -> `Learner`
  - 采样逻辑与固定形态类似，但用 `MaTrajBatchDisc` 支持图结构/list 状态。
  - 会在采样时收集部分 `design_params` 并写入当前 run 目录的 `0.csv` / `1.csv`。

## 4. 对手采样策略

- `use_opponent_sample=False`、评估阶段、或 `epoch==0` 时，双方一般使用当前 epoch 策略。
- `use_opponent_sample=True` 时，对每个 player 单独采样：
  - 当前 player 使用当前 epoch checkpoint；
  - opponent 从 `[floor(epoch * delta), epoch]` 中随机采样历史 checkpoint；
  - runner 分别采 agent0 视角和 agent1 视角的数据，再抽取各自 ego 数据更新。
- 这与论文中的 `delta-Uniform opponent sampling` 对齐。

## 5. 网络结构

### 5.1 固定形态 NormalPolicy / NormalValue

- `NormalPolicy`
  - 输入是 agent observation。
  - RunningNorm -> 可选 pre-MLP -> control MLP -> action mean。
  - action distribution 是 diagonal Gaussian。
  - `control_action_log_std` 是可学习参数，除非配置 `fix_control_std`。

- `NormalValue`
  - RunningNorm -> 可选 pre-MLP -> MLP -> scalar value。

### 5.2 evolved morphology: Transform2ActPolicy / Transform2ActValue

- `Transform2ActPolicy` 按阶段分流：
  - `skel_trans`：骨架结构动作，Categorical，动作含 add/remove/none，受 `enable_remove` 影响。
  - `attr_trans`：形态属性动作，Gaussian，输出每个 body 的 design 参数调整。
  - `execution`：控制动作，Gaussian，输出每个非 torso 节点的控制量。
- 三个阶段可分别配置 GNN、MLP、index-conditioned MLP。
- `get_log_prob` 会按节点累加 action log prob，再还原为每个样本的 log prob。
- `Transform2ActValue`
  - 输入图状态，可选择把 design stage flag 拼进 critic 输入。
  - GNN 后用根/首节点 value 作为整图 value。

### 5.3 developed morphology: DevPolicy / DevValue

- `DevPolicy` 两阶段：
  - `attribute_transform`：根据 `scale_state` 输出 scale/design action。
  - `execution`：根据 sim observation 或 entire observation 输出控制动作。
- scale action 会 clamp 到 `[-1, 1]`。
- `DevValue` 将 `[stage_ind, scale_state, sim_obs]` 拼接后经 MLP 输出 value。

## 6. 损失函数与优化

- 三类 learner 都使用 PPO clipped objective：
  - `ratio = exp(log_prob_new - log_prob_old)`
  - `surr1 = ratio * advantage`
  - `surr2 = clip(ratio, 1 - eps, 1 + eps) * advantage`
  - policy loss = `-mean(min(surr1, surr2))`
- Advantage 用 `estimate_advantages`，即 GAE：
  - `delta_t = r_t + gamma * V_{t+1} * mask_t - V_t`
  - advantage 反向递推，并做标准化。
  - return = value + advantage。
- `EvoLearner`
  - 会更新 critic：MSE(`V`, return)。
  - policy/value 使用 Adam 或 SGD，默认 Adam。
  - policy grad clip max norm 40。
- `DevLearner`
  - 会更新 critic：MSE + value net L2 regularization。
  - 计算 entropy 但当前 entropy bonus 被注释掉。
  - policy grad clip max norm 40。
- `Learner`
  - 当前代码中 value update 调用被注释掉，实际只更新 policy。
  - 但 advantage 仍使用当前 value net 估计，因此固定形态训练路径这里可能与标准 PPO/论文期望存在偏离，需要后续确认这是否是有意改动。

## 7. 环境与 reward

- run-to-goal：
  - sparse reward 是到达对方身后 goal line 的 `+1000/-1000`。
  - dense reward 来自 agent 的 `after_step`，通常包含向目标方向速度、控制代价、接触代价、生存项等，具体随 agent 类不同。
  - `use_exploration_curriculum=True` 时 runner 使用 `reward_parse` 和 `reward_dense` 做退火混合。

- robo-sumo：
  - sparse reward 是胜负：胜者 `+2000`、败者 `-2000`，超时 draw penalty `-1000`。
  - dense reward 包含 alive、ctrl、move_to_opp、push_opp 等。
  - arena 半径可随 version 变化，但当前采样调用中是否传 version 需要继续确认。

- evo env：
  - `MultiEvoAgentEnv` 每 episode 从 `skeleton_transform` 开始，然后 `attribute_transform`，最后 `execution`。
  - skeleton/attribute 阶段 reward 为 0，execution 阶段通过物理交互获得 dense/sparse reward。
  - XML 会在形态动作后重建并重新加载 MuJoCo env。

- dev env：
  - `MultiDevAgentEnv`/`RoboSumoDevEnv` 每 episode 从 `attribute_transform` 开始，然后 `execution`。
  - 通过 XML 缩放固定拓扑形态参数，不做骨架 add/remove。

## 8. 与论文描述的初步差异或待确认点

- 论文主要描述 fixed-topology 参数编码的 evo-ant/evo-bug/evo-spider；当前仓库还包含 Transform2Act 式 skeleton add/remove 的 `evo_ant` 路径，以及 developed morphology 的 `dev_*` 路径。
- 配置中 `policy_lr` 多为 `5e-5`，论文实验段写 Adam learning rate `0.0005`，需确认当前分支是否有意使用更小学习率。
- 固定形态 `Learner` 当前未实际更新 critic，这可能影响 run-to-goal/robo-sumo fixed morph baseline。
- `ind = exps.nonzero(...)` 在 `Learner`/`EvoLearner` 的非 mini-batch 分支中计算后未使用；当前常用配置启用 mini-batch，所以影响有限，但值得记录。
- `Config.out_dir` 写死为 `/root/ws/competevo/tmp`，本地环境运行可能需要额外处理日志输出路径。
- 当前工作区已有多个未提交改动，后续修改需要避免覆盖这些已有改动。

## 9. 当前讨论问题的理解

### 9.1 PPO、输入输出、形态动作频率与采样

当前训练策略是 PPO。更准确地说，是两个 agent 分别维护自己的策略网络与价值网络，各自采样轨迹、各自用 PPO 更新。它不是中心化 critic，也不是 population-based evolution；形态演化本身被看成策略输出的一部分，通过 RL 的回报信号优化。

固定形态时：

- 网络输入是当前机器人的观测，主要包括自身关节/根节点位置 `qpos`、关节速度 `qvel`、对手位置或相对位置；在 robo-sumo/fighter 类环境中还会加入接触力、torso 姿态矩阵等。
- 网络输出不是“关节角度目标”本身，而是 MuJoCo actuator control。可以理解为发送给电机/关节 actuator 的连续控制量，具体物理含义取决于 XML 中 actuator 的定义。
- 输出频率是每个环境 step 一次，也就是执行阶段每一步都输出控制动作。

有形态优化时，策略仍然是一个整体策略，但它按阶段输出不同语义的动作：

- `dev_*` developed morphology：
  - 每个 episode 开始先进入 `attribute_transform`。
  - 策略输出一次 scale/design action，用来缩放腿长、腿粗、gear 等固定拓扑参数。
  - 环境据此重建 XML，然后进入 `execution`。
  - execution 阶段每个 step 输出 actuator control。

- `evo_*` evolved morphology：
  - 每个 episode 开始先进入 `skeleton_transform`，持续 `skel_transform_nsteps` 次。
  - skeleton action 是离散动作，例如对某个 body 选择 none/add/remove；是否允许 remove 由配置决定。
  - 然后进入 `attribute_transform`，通常执行一次，输出连续形态参数修改。
  - 随后进入 `execution`，每个 step 输出 actuator control。

所以，形态参数不是每个物理控制步都重新输出，而是每局开始输出一次或少数几次；控制动作是在对抗执行阶段高频输出。论文里的说法是：每局开始生成 morph `M`，然后用这个 morph 进行 arena confrontation。

训练时动作采样是随机的：

- 连续控制/连续形态参数来自 Gaussian distribution。
- skeleton 结构动作来自 Categorical distribution。
- 评估或 display 时通常使用 mean action / argmax action。

这意味着形态本身也有采样性：同一个策略 checkpoint 在训练时可以采出不同形态；评估时则更接近使用策略均值生成的确定性形态。

### 9.2 不同训练阶段的奖励函数及含义

需要把三个概念分开：

1. warm-up 训练阶段。
   - 论文思想：先让 agent 学会基础运动能力，例如稳定、低能耗、朝正确方向快速移动。
   - 这一阶段的奖励主要是 dense reward。
   - 在 run-to-goal 中，dense reward 主要鼓励朝目标方向前进，同时惩罚控制能耗、接触代价或不稳定。
   - 在 sumo 中，dense reward 主要鼓励保持存活、靠近对手、把对手推离中心，同时惩罚控制能耗。
   - 当前代码没有自动 warm-up 调度；如果使用两阶段训练，经验通过 checkpoint 参数继承进入第二阶段。

2. confrontation 对抗训练阶段。
   - 目标从“会走、会接触”转向“赢”。
   - sparse reward 开始变重要。
   - run-to-goal 的 sparse reward：先到达对方身后 goal line 的 agent 获胜，胜者 `+1000`，败者 `-1000`。
   - robo-sumo 的 sparse reward：把对手推出擂台或打倒，胜者 `+2000`，败者 `-2000`；超时平局有惩罚。

3. reward annealing。
   - 论文公式：`R = kappa * R_dense + (1 - kappa) * R_sparse`。
   - 代码中对应 `reward_dense` 和 `reward_parse`。
   - 早期 `kappa` 大，训练主要学习基本技能；后期 `kappa` 降到 0，训练主要由胜负 reward 驱动。

因此，你的理解“两个阶段网络一样，分阶段靠参数继承”在一个前提下成立：warm-up 和 confrontation 使用同一类 agent、同一套可兼容网络配置。此时第二阶段不是换网络，而是继承第一阶段的 policy/value 参数继续 PPO。若第一阶段固定形态、第二阶段可变形态，则网络结构不同，不能简单认为是同一个网络继续训练。

对于形态优化任务，morph-generation 阶段即时 reward 为 0，但它不是无监督的。形态动作被存在同一条 episode 轨迹中，后续 execution 阶段赢/输、跑得快/慢、推得动/推不动都会通过 return/advantage 回传给一开始的形态动作。这就是“身体参数为什么能被 PPO 优化”的关键。

### 9.3 两个智能体的策略网络与 delta-Uniform 对手采样

在 `multi-agent-runner` 和 `multi-evo-agent-runner` 下，两个智能体不是共享一个策略网络；agent0 和 agent1 各自有 learner、policy、value、checkpoint 目录。它们是两个独立训练体。

论文中的 self-practice / delta-Uniform opponent sampling 想解决的问题是：如果两个当前策略永远同步互打，训练容易不稳定；如果对手太强或太弱，学习信号也差。因此，每次训练某个 agent 时，让它面对对手历史策略池中的随机旧版本。

当前代码里的逻辑可以概括为：

- 训练 agent0 时：
  - agent0 使用自己的最新策略。
  - agent1 从 agent1 的历史 checkpoint 中按 delta-Uniform 采样一个旧策略作为对手。
  - 这批数据主要用于更新 agent0。

- 训练 agent1 时：
  - agent1 使用自己的最新策略。
  - agent0 从 agent0 的历史 checkpoint 中按 delta-Uniform 采样一个旧策略作为对手。
  - 这批数据主要用于更新 agent1。

采样的是历史策略参数，不是直接采样某个历史形态向量。对于可变形态 agent，历史策略参数里包含 morph sub-policy，所以被采中的旧对手会在 episode 开始重新生成自己的形态。

delta 的含义：

- `delta` 控制历史池的起点。
- 代码中近似为从 `[floor(epoch * delta), epoch]` 附近的历史 checkpoint 中抽对手。
- `delta` 越小，对手池越宽，可能抽到更早、更弱、更多样的对手。
- `delta` 越大，对手池越靠近当前 epoch，对手更接近当前水平。

因此，“训练体是最新策略，对手是历史策略采样”这个理解基本正确，但要补充一点：两个 agent 都会轮流作为训练体；不是永远 agent0 是训练体、agent1 是历史对手。

### 9.4 当前算法超参数参照

下表按 `Config` 默认值和常用配置值整理。`Config 默认值` 是 YAML 没写时程序会采用的值；`常用配置值` 以当前 run-to-goal / robo-sumo 配置为主。

| 超参数 | Config 默认值 | 常用配置值 | 含义 |
| --- | --- | --- | --- |
| `gamma` | `0.99` | `0.995` | 折扣因子。越接近 1，越重视长期回报。 |
| `tau` | `0.95` | `0.95` | GAE 参数。控制 advantage 估计的偏差/方差折中。 |
| `clip_epsilon` | `0.2` | `0.2` | PPO clip 范围，限制新旧策略概率比变化过大。 |
| `policy_optimizer` | `Adam` | `Adam` | actor/policy 优化器。 |
| `policy_lr` | `5e-5` | `5e-5` | 策略网络学习率。论文实验段写 `0.0005`，当前配置更小。 |
| `policy_momentum` | `0.0` | `0.0` | SGD 时使用；Adam 时基本不用。 |
| `policy_weightdecay` | `0.0` | `0.0` | 策略网络 weight decay。 |
| `value_optimizer` | `Adam` | `Adam` | critic/value 优化器。 |
| `value_lr` | `3e-4` | `3e-4` | 价值网络学习率。 |
| `value_momentum` | `0.0` | `0.0` | SGD value optimizer 时使用。 |
| `value_weightdecay` | `0.0` | `0.0` | value 网络 weight decay。 |
| `l2_reg` | `1e-3` | dev 配置 `1e-3` | `DevLearner` 更新 value 时额外加入的 L2 正则。 |
| `num_optim_epoch` | `10` | `10` | 每批 rollout 数据重复做多少轮 PPO 更新。 |
| `min_batch_size` | `50000` | `50000` | 每个 epoch 至少采样多少 step。 |
| `mini_batch_size` | `min_batch_size` | `2048` | PPO mini-batch 大小。小于 `min_batch_size` 时启用 mini-batch 更新。 |
| `eval_batch_size` | `10000` | 多数未显式写，默认 `10000` | 每个 epoch 评估采样步数。 |
| `max_epoch_num` | `1000` | run-to-goal 多为 `1000`，robo-sumo fixed 为 `2000` | 最大训练 epoch/generation 数。 |
| `seed` | `1` | 配置中常见 `1/3/42` | 随机种子。 |
| `save_model_interval` | `100` | 常用 `1` | 每隔多少 epoch 保存一次 checkpoint。 |
| `use_reward_scaling` | `False` | `False` | 是否对 reward 做 running scaling。 |
| `use_opponent_sample` | `False` | 对抗配置多为 `True` | 是否使用历史对手策略采样。 |
| `delta` | `0.0` | run-to-goal/evo/dev 常见 `0.5`，robo-sumo fixed 常见 `1` | 历史对手池起点比例。小则对手更多样，大则对手更接近当前。 |
| `use_exploration_curriculum` | `False` | 对抗配置多为 `True` | 是否启用 dense/sparse reward 退火。 |
| `termination_epoch` | `200` | run-to-goal fixed `200`，dev/evo 常见 `1000`，robo-sumo fixed `2000` | reward annealing 结束时刻。 |
| `use_parse_reward` | `True` | warm-up/single-agent 可能 `False`，对抗多为 `True` | 是否启用胜负稀疏奖励。 |
| `policy_specs.control_log_std` | 无统一默认，需配置 | 常用 `0` | Gaussian control action 的初始 log std。 |
| `fix_control_std` | 无统一默认，需配置 | 常用 `false` | 是否固定控制动作标准差。 |
| `attr_log_std` | 仅 evo 配置 | `-2.3` | evo attribute action 的初始 log std，较小表示形态参数采样更保守。 |
| `robot_param_scale` | `0.1` | evo 配置常用 `1` | 形态动作映射到实际参数变化的尺度。 |
| `skel_transform_nsteps` | `5` | `run-to-goal-evoants-v0` 当前为 `1` | 每局开始执行多少次 skeleton transform。 |
| `enable_remove` | `True` | `run-to-goal-evoants-v0` 为 `false` | skeleton action 是否允许删除 body。 |
| `max_body_depth` | `4` | evo 配置常见 `5` | 可生成/保留身体树的最大深度。 |
| `min_body_depth` | `1` | 默认 `1` | 允许形态变换的最小身体深度。 |
| `add_body_condition.max_nchild` | 配置 dict 默认空 | evo 配置常见 `5` | 限制某个 body 最多能有多少 child。 |
| `use_entire_obs` | `False` | dev 配置多为 `False` | dev 控制分支是否使用完整 `[stage, scale, sim_obs]`；否则只用 `sim_obs`。 |
| `num_threads` | CLI 默认 `1` | 用户运行时指定 | 采样进程数。论文提到 50 parallel rollouts，当前 CLI 默认较小。 |
| `use_cuda` | CLI 默认 `True` | 取决于机器 | 是否尝试使用 GPU。 |

### 9.5 为什么 agent0 和 agent1 要轮流训练？

在当前 `multi-agent-runner` / `multi-evo-agent-runner` 设计中，agent0 和 agent1 是两条独立学习线：各自有策略网络、价值网络、optimizer、checkpoint。轮流训练的意义是让双方都持续适应，而不是让一个固定成为“陪练靶子”。

只训练 agent0、让 agent1 从 agent0 历史中采样，在某些严格对称的 self-play 场景是可以成立的，代码里也有 `selfplay-agent-runner` 这种思路。但它不适合作为当前 CompetEvo 多数实验的默认范式，原因是：

- CompetEvo 关注的是双方在竞争中共同提升，轮流训练更接近论文的 `P_alpha` 和 `P_beta` 两个策略池。
- 很多任务不是完全同分布：左右出生点、朝向、species、fixed/dev/evo 形态都可能不同。
- 如果只训练 agent0，agent1 没有自己的适应过程，实验就变成“一个 agent 打自己的旧版本”，而不是“两类玩家/两条谱系共同演化”。
- 对抗训练需要强弱合适的对手。轮流训练并从对方历史采样，可以减少策略崩坏、过拟合单一对手、或一方过强导致另一方没有学习信号的问题。
- 对 asymmetric species 或 fixed-vs-evolved 比较，agent0 历史不能代表 agent1 的物种/形态/策略分布。

因此，是否只训练一个 agent 取决于实验目的：若目标是单策略 self-play，可以考虑 `SPAgentRunner`；若目标是 CompetEvo 式双玩家共同演化，应保留 agent0/agent1 轮换训练。

### 9.6 episode 内形态不变，但形态策略网络是否已经改变？

这里要区分 rollout 时序和 PPO 更新时序。

- 在一个 episode 内，策略网络参数不会更新。runner 只是用当前 checkpoint/当前 learner 参数采样动作。
- 对可变形态 agent，episode 开头先采样形态动作并重建 XML；进入 execution 后，这一局身体固定，不会边打边长。
- 当前 episode 结束后也不会立刻更新网络；runner 会收集到至少 `min_batch_size` 的 rollout 数据。
- PPO 更新发生在一个 epoch 的采样批次结束之后。更新时，形态分支和控制分支都会根据同一批 return/advantage 调整参数。

所以，更准确的说法是：同一个 episode 中，形态和形态策略网络都不变；一个 epoch 更新完成后，形态策略网络会变，下一批 episode 开始时可能生成新的形态。训练时形态动作本身是随机采样的，因此同一策略参数在不同 episode 也可能采到不同形态；评估时用 mean action 会更确定。

### 9.7 是否已有 run-to-goal checkpoint？如何先 train run-to-goal 再 train 对抗？

当前仓库 `runs/` 下只发现示例 checkpoint：

- `runs/robo-sumo-devants-v0/models/agent_0/best.p`
- `runs/robo-sumo-devants-v0/models/agent_1/best.p`

没有发现仓库内自带的 run-to-goal checkpoint。

在代码层面，“先训练 run-to-goal，再训练对抗”的机制是 checkpoint 继承，而不是 replay buffer 迁移。基本流程是：

1. 第一阶段训练 run-to-goal / warm-up：

```bash
conda run -n EAI python train.py --cfg config/run-to-goal-ants-v0.yaml
```

训练会输出到 `./tmp/<env_name>/<timestamp>/models/agent_0/` 和 `agent_1/`。

2. 第二阶段训练目标对抗任务，并从第一阶段 checkpoint 初始化：

```bash
conda run -n EAI python train.py \
  --cfg config/robo-sumo-ants-v0.yaml \
  --ckpt_dir ./tmp/run-to-goal-ants-v0/<timestamp>/models \
  --ckpt best
```

但这里有一个非常重要的兼容性条件：第二阶段的 agent 类型、观测维度、动作维度、网络结构必须和第一阶段 checkpoint 兼容，否则 `load_state_dict` 会因为 shape 不一致失败，或者语义上不可靠。

需要特别注意：

- `run-to-goal-ants-v0` 使用的是普通 ant；`robo-sumo-ants-v0` 使用的是 robo/fighter ant，观测和 reward 设计不同，checkpoint 未必兼容。
- `run-to-goal-devants-v0` 使用 `dev_ant`；`robo-sumo-devants-v0` 使用 `dev_ant_fighter`，观测也可能不同。
- 固定形态 checkpoint 不能直接初始化 `dev_*` 或 `evo_*` 网络，因为网络结构不同。

如果目标是论文意义上的 warm-up，比较稳妥的代码层面做法是：为第二阶段同一个 agent 类型设计一个 warm-up 配置，只改变 reward 或 `use_parse_reward`，尽量保持 observation/action/network specs 一致。这样第一阶段 checkpoint 才能自然继承到第二阶段。

训练 run-to-goal 的过程是否两个智能体，取决于环境 id：

- `run-to-goal-ant-v0`、`run-to-goal-bug-v0`、`run-to-goal-spider-v0`、`run-to-goal-evoant-v0` 是单 agent 注册。
- `run-to-goal-ants-v0`、`run-to-goal-bugs-v0`、`run-to-goal-spiders-v0`、`run-to-goal-devants-v0`、`run-to-goal-evoants-v0` 是双 agent 注册。

双 agent 的 `MultiAgentRunner` / `MultiEvoAgentRunner` 默认不共享权重。agent0 和 agent1 架构通常相同，但参数独立，分别保存在 `models/agent_0/` 和 `models/agent_1/`。如果想共享权重，需要改 runner/learner 的构造方式，目前默认不是共享策略。

### 9.8 checkpoint 加载预训练结果的真实机制

当前代码加载 checkpoint 是严格的同构网络加载；并没有自动解决 run-to-goal 和对抗任务观测不同的问题。

加载链路：

- `train.py` 解析 `--ckpt_dir` 和 `--ckpt`。
- `BaseRunner.__init__` 在创建 env 和 learner 后调用 `load_checkpoint`。
- `MultiAgentRunner.load_agent_checkpoint` / `MultiEvoAgentRunner.load_agent_checkpoint` 从 `ckpt_dir/agent_i/<ckpt>.p` 读取 pickle。
- learner 的 `load_ckpt` 直接调用：
  - `self.policy_net.load_state_dict(model['policy_dict'])`
  - `self.value_net.load_state_dict(model['value_dict'])`

这里没有 `strict=False`，也没有过滤 shape 不一致的层。因此预训练 checkpoint 只有在目标 env 构造出的 policy/value 网络结构完全兼容时才能加载。

对 `dev` agent 的 shape 检查结果：

```text
config/run-to-goal-devants-v0.yaml DevAnt sim_obs_dim 31 state_dim 52
config/robo-sumo-devants-v0.yaml DevAntFighter sim_obs_dim 118 state_dim 139

policy 不兼容层:
control_norm.mean                    (31,)    -> (118,)
control_norm.var                     (31,)    -> (118,)
control_norm.std                     (31,)    -> (118,)
control_mlp.affine_layers.0.weight   (64,31)  -> (64,118)

value 不兼容层:
norm.mean                            (52,)    -> (139,)
norm.var                             (52,)    -> (139,)
norm.std                             (52,)    -> (139,)
mlp.affine_layers.0.weight           (64,52)  -> (64,139)
```

这说明：

- `dev` 形态分支的 `scale_mlp` 和 `scale_state_mean` 理论上可以迁移，因为 `scale_state_dim=20` 一致。
- control 分支不能直接迁移，因为 robo-sumo observation 比 run-to-goal 多了 contact force、torso matrix 等信息。
- value 网络也不能直接迁移，因为 critic 输入是 `[stage_ind, scale_state, sim_obs]`，`sim_obs` 变化导致 `state_dim` 变化。

因此，如果目标是“run-to-goal 预训练 -> 对抗训练”并且不改加载代码，就必须保证两个阶段的 observation/action/network shape 一致。否则需要显式实现 partial loading，并清楚声明哪些层继承、哪些层重新初始化。

## 10. 本轮实验闭环记录（2026-06-01）

本轮任务采用“同网络，不同目标函数”的兼容方案：不再把 `run-to-goal-devants-v0` 的 checkpoint 直接加载到 `robo-sumo-devants-v0`，而是在 `robo-sumo-devants-v0` 的 fighter 观测和网络结构下增加 `reward_specs.mode: run_to_goal_warmup`，先训练向对手移动的 dense reward，再切换到 sumo sparse+dense reward。

已完成的短训练 sanity check：

- 固定形态：`config/repro/basic-run-to-goal-ants-sanity.yaml`
  - run dir: `tmp/run-to-goal-ants-v0/20260601_211152`
  - final checkpoint: `models/agent_0/epoch_0002.p`、`models/agent_1/epoch_0002.p`
- 形态可进化 warm-up：`config/repro/devants-compatible-warmup-sanity.yaml`
  - run dir: `tmp/robo-sumo-devants-v0/20260601_211201`
  - intermediate checkpoint: `models/agent_0/epoch_0002.p`、`models/agent_1/epoch_0002.p`
- 形态可进化对抗：`config/repro/devants-confrontation-sanity.yaml`
  - 从 warm-up `epoch_0002` 加载，严格同构加载成功
  - run dir: `tmp/robo-sumo-devants-v0/20260601_211212`
  - copied start checkpoint: `models/agent_*/epoch_0000.p`
  - final checkpoint: `models/agent_*/epoch_0002.p`
- runs 短复现：`config/repro/reproduce-robo-sumo-devants-sanity.yaml`
  - run dir: `tmp/robo-sumo-devants-v0/20260601_211230`
  - final checkpoint: `models/agent_*/epoch_0002.p`

评测与报告输出：

- HTML 报告：`reports/task_experiment_report.html`
- 评测 JSON：`reports/eval/*.json`
- 训练曲线：`reports/curves/*.svg`

关键评测结果（5 episodes，mean action）：

| 对象 | 平均回报 agent0 / agent1 | 胜率 agent0 / agent1 | 平局率 |
|---|---:|---:|---:|
| fixed run-to-goal sanity | 502.02 / 496.96 | 0.00 / 0.00 | 1.00 |
| dev warm-up sanity | 1076.43 / 1123.39 | 0.00 / 0.00 | 1.00 |
| dev confrontation sanity | -1087.79 / -1130.68 | 0.00 / 0.00 | 1.00 |
| runs original best | 3023.87 / -1318.11 | 0.80 / 0.00 | 0.20 |
| short reproduction epoch_0002 | -1168.99 / -2067.60 | 0.20 / 0.00 | 0.80 |

短复现没有达到 runs 原 checkpoint 的效果，主要原因是本轮只跑 2 epoch、`min_batch_size=128`，而原始训练配置使用长训练和大 batch。该结果只能说明训练/保存/加载/评测/画图流程可用，不能作为收敛复现实验结论。
