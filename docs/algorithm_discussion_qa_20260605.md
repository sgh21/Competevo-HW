# 算法与评测 Q&A 记录（2026-06-05）

本文记录本轮关于 warm-up、任意两智能体对打评测、reward、公平评价和后续算法改进的讨论结论。

## Q1：当前训练的五个任务都是用 run-to-goal 做 warm-up 吗？

不是，需要区分两个实验脚本口径。

`scripts/run_long_training_pipeline.sh` 口径下，脚本启动 4 段训练和 5 个评测：`fixed_run_to_goal` 是真正的 `run-to-goal-ants-v0` 固定形态训练；`devants_warmup` 是 `robo-sumo-devants-v0` 环境中的兼容 warm-up，配置为 `reward_specs.mode: run_to_goal_warmup`；`devants_confrontation` 是从 warm-up checkpoint 加载后切换到 `reward_specs.mode: sumo` 的对抗训练；`reproduction_robo_sumo_devants` 是直接跑 `config/robo-sumo-devants-v0.yaml` 的对抗复现；第 5 个是对仓库自带 `runs/robo-sumo-devants-v0/models` 的评测，不是训练。

`reports/unified_modes/run_formal_training_parallel.sh` 口径下，确实有 5 条 formal 训练线：`sp-fixed`、`tp-fixed`、`sp-morph`、`tp-morph`、`tp-mixed-a0morph`。这五条线都从同一个 warm-up checkpoint 初始化，但训练配置是 `config/repro/unified-devant-training.yaml`，`reward_specs.mode: sumo`。因此它们不是 warm-up 本身，而是 warm-up 之后的正式对抗训练。

## Q2：warm-up 时的观测是什么，是否和后续对抗训练一致？

对当前 `devants-compatible-warmup-long.yaml` 来说，观测与后续 `devants-confrontation-long.yaml` 是一致的。两者都使用 `env_name: robo-sumo-devants-v0`、`runner_type: multi-evo-agent-runner`、`DevAntFighter` agent，以及基本一致的 `dev_policy_specs` / `dev_value_specs`。

关键设计是“不换环境、不换 fighter 观测，只换 reward mode”。这避免了 `run-to-goal-devants-v0` 与 `robo-sumo-devants-v0` 观测维度不一致导致 checkpoint 无法严格加载的问题。

`DevAntFighter._get_obs(stage)` 返回三段 list：

```text
[stage_ind, scale_vector, sim_obs]
```

`stage_ind` 表示当前阶段，`attribute_transform` 为 0，`execution` 为 1；`scale_vector` 是 20 维形态缩放参数；`sim_obs` 是 fighter 物理观测，包括自身 `qpos`、`qvel`、裁剪后的接触外力、对手 torso 的位置，以及自身 torso 姿态矩阵。

策略侧默认 `use_entire_obs: false`，所以形态分支输入 `scale_vector`，输出 scale/design action；控制分支只使用 `sim_obs`，输出执行阶段 actuator control；critic/value 使用完整 `[stage_ind, scale_vector, sim_obs]`。

## Q3：warm-up 的输出是否包含形态动作？

包含，但要看 `morph_optim_agents` 的设置。

`DevPolicy` 每次输出完整动作向量：

```text
[scale_action, control_action]
```

其中 `scale_action` 长度为 `scale_state_dim`，`control_action` 长度为 `sim_action_dim`。环境按阶段解释动作：`attribute_transform` 阶段只使用前面的 `scale_action`，修改 XML 中的身体尺寸、位置偏移和 gear；`execution` 阶段只使用后面的 `control_action`，发送到 MuJoCo actuator。

在 formal 五任务里，`morph_optim_agents: none` 时，策略仍然有 scale action 这一段，但会被固定为 `fixed_morph_scale`，并且 `DevLearner` 会过滤掉形态阶段样本，只更新执行控制；`morph_optim_agents: all` 时，agent0/agent1 的形态动作都会被优化；`morph_optim_agents: 0` 时，只有 agent0 的形态分支被优化，agent1 的形态固定。

## Q4：任意两个智能体对打测评文件的输出是什么？

当前任意对打评测入口主要是 `scripts/evaluate_checkpoint.py`。它支持 `--agent0_ckpt_dir`、`--agent0_ckpt`、`--agent1_ckpt_dir`、`--agent1_ckpt`，可以让 agent0/agent1 分别来自不同 run 或不同 epoch。

示例：

```bash
conda run -n EAI python scripts/evaluate_checkpoint.py \
  --cfg <compatible_config.yml> \
  --ckpt_dir <default_models_dir> \
  --ckpt <default_ckpt> \
  --agent0_ckpt_dir <agent0_models_dir> \
  --agent0_ckpt <agent0_ckpt> \
  --agent1_ckpt_dir <agent1_models_dir> \
  --agent1_ckpt <agent1_ckpt> \
  --episodes 100 \
  --out reports/.../match.json
```

输出 JSON 包括 `checkpoint_files`、`avg_episode_reward`、`std_episode_reward`、`wins`、`win_rate`、`draws`、`draw_rate`、`rewards_by_episode`、`episode_lengths`、`avg_episode_length`。

需要注意：`evaluate_checkpoint.py` 直接累计环境 `env_rewards`。如果配置启用了 `use_exploration_curriculum`，这里并不会调用 runner 的 `custom_reward()` 退火逻辑，因此 JSON 里的 reward 更接近环境原始 `reward_parse + reward_dense`。胜率和平局率通常比 reward 更适合作为对抗强弱指标。

## Q5：现在是否可以查看对打视频、双方胜率和 reward？

可以查看胜率和 reward：`evaluate_checkpoint.py` 会把这些指标写到 JSON，并同步打印到终端。

可以实时观看对打：`display.py` 现在支持 per-agent checkpoint 参数：

```bash
conda run -n EAI python display.py \
  --cfg <compatible_config.yml> \
  --ckpt_dir <fallback_models_dir> \
  --agent0_ckpt_dir <agent0_models_dir> \
  --agent0_ckpt <agent0_ckpt> \
  --agent1_ckpt_dir <agent1_models_dir> \
  --agent1_ckpt <agent1_ckpt>
```

但目前还不能直接保存视频文件。`display.py` 使用 `render_mode="human"` 做窗口渲染，runner 的 `display()` 会在日志里输出平均 reward 和胜率；如果要生成 mp4/gif，需要新增一个 video recording 入口，用 `render_mode="rgb_array"` 抓帧并通过 `imageio` 或 Gymnasium wrapper 写出。

## Q6：当前策略如何评价智能体行为好坏？

训练层面，策略好坏由 PPO 的 advantage/return 决定。runner 采样 rollout；learner 用 reward、mask、value 估计 GAE advantage；policy 通过 PPO clipped objective 最大化 advantage 对应动作的概率；critic 学习预测 return；每个 epoch 后用 mean action 评估 reward、win rate、episode length，并决定是否保存 best checkpoint。

所以“行为好坏”不是人工规则直接打分，而是由环境 reward 经过 return/advantage 转换后反馈给策略。

## Q7：对抗过程中的 reward 是什么？

以 `robo-sumo-devants-v0` 为例，环境 reward 分成 sparse 和 dense 两类。

Sparse / `reward_parse`：对手倒地或出擂台，自己 `+2000`；自己倒地或出擂台，自己 `-2000`；到最大步数仍未分胜负，draw penalty `-1000`；warm-up 模式或 `use_parse_reward: false` 时，这部分被置为 0。

Dense / `reward_dense`：`alive_reward = +2.0`；`ctrl_reward = -0.1 * sum(action^2)`；`move_to_opp_reward` 是朝对手方向移动的速度投影，乘以系数 10；`push_opp_reward` 与对手离中心距离相关，离中心越远惩罚越小，sumo 模式加入，`run_to_goal_warmup` 模式不加入。

在 `use_exploration_curriculum: true` 时，runner 实际训练 reward 是：

```text
alpha * reward_dense + (1 - alpha) * reward_parse
alpha = max((termination_epoch - epoch) / termination_epoch, 0)
```

因此早期更重视会动、能靠近对手、保持稳定；后期更重视真实胜负。

## Q8：当前策略评价公平吗？单看训练曲线意义大吗？

单看训练曲线有参考价值，但不足以作为“绝对实力”评价。

原因是对抗训练是非平稳的：同一个 agent 的 reward 不仅取决于自身策略，也取决于本轮遇到的对手、对手 checkpoint 年龄、出生侧、形态是否优化、是否使用 mean action 等。对手弱，reward 和 win rate 可能虚高；对手突然变强，曲线可能下降，但这不一定代表自己退步。

因此更公平的评价应该使用固定评测协议：固定一组候选 checkpoint；每两个候选都进行 round-robin 对战；每个 pair 同时评估 A as agent0 vs B as agent1，以及 B as agent0 vs A as agent1；使用相同 episode 数、相同 seed 列表、mean action；输出胜/平/负矩阵、平均 reward、平均 episode length、Elo/TrueSkill 或积分榜；对随机策略鲁棒性可额外跑 stochastic action 版本。

如果“五个任务”指 formal 五条训练线，那么做积分赛非常有意义，因为它们都在 `robo-sumo-devants-v0` 同构环境和网络下，可以直接比较。如果“五个任务”混合了 run-to-goal、warm-up、sumo、fixed ant 和 dev ant，则需要先保证 env、agent 类型、obs/action 维度和 checkpoint 结构兼容，否则不能直接放进同一个积分赛。

## Q9：如何改进训练以获得更强智能体？

优先建议从“对手选择”和“评测闭环”改起，而不是先大改模型。

1. 建立 league / round-robin 评测池。每隔固定 epoch 把候选 checkpoint 放进评测池，做小规模积分赛。训练时的 best 不只看当前 eval reward，也参考固定池胜率或 Elo，避免保存到只擅长打当前对手的 checkpoint。

2. 优化 opponent sampling。当前 `use_opponent_sample=True` 时，对手从 `[floor(epoch * delta), epoch]` 均匀采样。可以采样胜率接近 40%-60% 的对手，让学习信号更强；混合 recent / best / historical opponent；对刚打不过的对手提高采样概率；对过弱或过强的对手降低采样概率；用 Elo/TrueSkill 估计对手强度，采样 rating 相近或略强的对手。

3. 调整 `delta`。`delta` 越大，对手越接近当前版本，训练更像势均力敌的近邻对抗；`delta` 越小，对手池更宽，抗遗忘更好但可能包含太多弱对手。可以从 `0.5`、`0.75`、`0.9` 做 ablation。

4. 保留少量 dense reward。当前 curriculum 到 `termination_epoch` 后会退火到纯 sparse reward。对于 sumo 这种稀疏胜负任务，可以考虑保留一个 dense floor，例如 `0.1 * dense + 0.9 * sparse`，减少策略在后期失去移动/接触 shaping 信号。

5. 增加 entropy bonus 或动作噪声调度。`DevLearner` 中 entropy bonus 目前被注释。可以加一个小的 entropy coefficient 并随 epoch 衰减，减少早期策略过快塌缩到单一动作/单一形态。

6. 形态优化分阶段。对 morph 任务，可以先固定形态只训控制，再只放开 agent0 或双方形态，最后联合微调。这样可以避免形态和控制同时漂移太快，导致双方都学不到稳定对抗技能。

7. 评测时强制 side-swap。环境 reset 已经有左右出生随机化，但正式积分赛仍建议做 A/B 两个方向的显式互换，避免某个 checkpoint 对出生侧或 agent id 有偏置。

8. 检查固定形态 PPO 的 critic 更新。`Learner` 路径里 value update 当前有被注释的迹象。若 fixed ant baseline 也要纳入严肃比较，建议单独确认并修正这一点，否则 fixed baseline 的训练质量可能被低估。

## Q10：下一步最值得做什么？

建议下一步先实现一个统一的积分赛评测脚本，而不是继续只看单条训练曲线。

目标入口可以支持若干候选名、cfg、agent0/agent1 checkpoint 路径；自动跑 all-vs-all 和 side-swap；输出 `match_matrix.json`、`leaderboard.csv`、简洁 HTML/Markdown 表格；指标包括 win/draw/loss、win rate、reward mean/std、episode length、Elo/积分；可选为指定 pair 生成 `rgb_array` 视频。

这样后续训练改动，例如调 `delta`、加 league sampling、改 reward curriculum，都能通过同一套评测协议比较，结论会比单看 TensorBoard 曲线稳很多。
