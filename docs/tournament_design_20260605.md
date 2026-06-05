# 积分赛规则与实现要求（2026-06-05）

本文记录 5 条 formal 训练线形成 8 个候选智能体后的积分赛设计，用于比较不同训练范式下智能体的对抗能力。

## 1. 参赛智能体定义

当前 5 条训练线应拆成 8 个独立参赛者：

| 候选类型 | 候选数量 | 来源说明 | 评测时形态动作 |
| --- | ---: | --- | --- |
| `sp-fixed` | 1 | 形态不优化 self-play，selfplay runner 的单 checkpoint | 固定形态，不启用 scale branch |
| `tp-fixed` | 2 | 形态不优化 two-player，对应 `agent_0` 和 `agent_1` | 固定形态，不启用 scale branch |
| `sp-morph` | 1 | 形态优化 self-play，selfplay runner 的单 checkpoint | 启用形态动作 |
| `tp-morph` | 2 | 双方形态都优化 two-player，对应 `agent_0` 和 `agent_1` | 启用形态动作 |
| `tp-mixed-a0morph` | 2 | agent0 形态优化、agent1 形态不优化的 two-player | agent0 候选启用形态动作，agent1 候选固定形态 |

建议使用以下稳定 ID：

```text
sp_fixed
tp_fixed_a0
tp_fixed_a1
sp_morph
tp_morph_a0
tp_morph_a1
tp_mixed_a0_morph
tp_mixed_a1_fixed
```

## 2. 来源追踪

积分赛必须先生成 `agents.yml` 或 `agents.json`，记录每个参赛者来源。每个参赛者至少包含：

- `agent_id`：积分赛内部唯一 ID；
- `display_name`：报告展示名；
- `training_line`：来自 `sp-fixed`、`tp-fixed`、`sp-morph`、`tp-morph`、`tp-mixed-a0morph` 中哪一条；
- `training_mode`：`selfplay` 或 `two_player`；
- `source_role`：selfplay 单策略、`agent_0`、`agent_1`；
- `run_dir`：训练输出目录；
- `checkpoint`：例如 `best`、`epoch_1000`；
- `checkpoint_file`：实际加载的 `.p` 文件；
- `morph_enabled`：该参赛者评测时是否允许输出形态动作；
- `fixed_morph_scale`：形态固定时采用的 scale，当前建议 `0.0`；
- `train_cfg`：原始训练 cfg 路径；
- `notes`：补充说明，例如 selfplay checkpoint 是否在根 `models/` 下。

## 3. 基础赛制

采用全循环双回合积分赛。8 个参赛者共有 `8 * 7 / 2 = 28` 个 unordered pair。每个 pair 跑两个 fixture：

1. A 放到环境 `agent0`，B 放到环境 `agent1`；
2. B 放到环境 `agent0`，A 放到环境 `agent1`。

总计 56 个 fixture。正式评测建议每个 fixture 跑 `50` 或 `100` episodes；调试时先跑 `5` 或 `10` episodes smoke test。

双回合用于抵消 agent id、出生侧、训练时角色、XML 命名和策略对 slot 的偏置。正式排名不能只看单向 A-as-agent0 vs B-as-agent1。

## 4. 初始化边与随机性

需要同时处理两类偏置：

- 左右出生边偏置：某些策略可能在左侧或右侧更强；
- agent slot 偏置：某些 checkpoint 原本是以 `agent_0` 或 `agent_1` 训练出来的。

建议规则：

1. 每个 fixture 使用固定 seed，可复现；side-swap 的两个 fixture 使用成对 seed。
2. 每局 reset 后记录双方初始 `x` 坐标或 left/right side。
3. 报告中展示每个参赛者在 left/right 两侧的胜率与平均 reward。
4. 若某个 fixture 左右分布明显不均衡，例如 100 局中某参赛者少于 40 局在某一侧，需要在报告中标注，必要时增加 episode 数。
5. 每个 pair 的最终结果必须聚合两个 slot-swap fixture。

## 5. 形态开关规则

这是积分赛最容易误评的地方。

当前代码中的 `morph_optim_agents` 按环境 slot 生效，而不是按 checkpoint 身份生效。因此积分赛脚本需要为每个 fixture 生成临时 eval cfg：

- 若当前放在 `agent0` slot 的参赛者 `morph_enabled=true`，则 `morph_optim_agents` 包含 `0`；
- 若当前放在 `agent1` slot 的参赛者 `morph_enabled=true`，则 `morph_optim_agents` 包含 `1`；
- 固定形态参赛者所在 slot 不出现在 `morph_optim_agents` 中，并使用 `fixed_morph_scale: 0.0`。

示例：`sp_morph` 对 `tp_fixed_a1`，若第一回合 `sp_morph` 在 slot 0、`tp_fixed_a1` 在 slot 1，则临时 cfg 应为：

```yaml
runner_type: multi-evo-agent-runner
reward_specs:
  mode: sumo
use_parse_reward: true
use_exploration_curriculum: false
morph_optim_agents: [0]
fixed_morph_scale: 0.0
```

第二回合换位后应变为：

```yaml
morph_optim_agents: [1]
```

这样才能保证形态优化智能体使用自己的形态策略，形态不优化智能体保持固定形态。

## 6. 积分规则

主排名采用 episode 级积分：

- 胜一局：`3` 分；
- 平一局：`1` 分；
- 负一局：`0` 分。

一个 fixture 有 N 局，则双方分别累计 N 局积分。一个 pair 的最终结果是两个 side-swap fixture 的合并结果。

排行榜字段：

- `rank`；
- `agent_id` / `display_name`；
- `points`；
- `points_per_episode`；
- `episodes`；
- `wins` / `draws` / `losses`；
- `win_rate` / `draw_rate` / `loss_rate`；
- `avg_reward`；
- `avg_reward_margin`；
- `avg_episode_length`；
- `agent0_win_rate` / `agent1_win_rate`；
- `left_win_rate` / `right_win_rate`；
- `elo` 或 `rating`，可选，只作为辅助指标。

排序建议：先按 `points`，再按 `win_rate`，再按 `avg_reward_margin`，最后按 head-to-head 积分。

## 7. 逐局与逐 fixture 记录

每个 fixture 输出一个独立 JSON：

```text
reports/league/<run_id>/matches/<fixture_id>.json
```

内容至少包括：

- fixture 基本信息：`fixture_id`、`pair_id`、`seed`、`episodes`；
- 参赛者和 slot：`slot0_agent_id`、`slot1_agent_id`、`slot0_checkpoint_file`、`slot1_checkpoint_file`；
- 临时 cfg：`eval_cfg_file`、`morph_optim_agents`；
- 汇总指标：双方 wins/draws/losses、win_rate、draw_rate、avg_reward、std_reward、avg_reward_margin、avg_episode_length；
- 逐局记录：每局 winner、双方 reward、episode length、双方初始 side、是否 truncated、终止原因；
- 可选 reward 分量：`reward_parse`、`reward_dense`、`win_reward`、`lose_penalty`、`ctrl_reward`、`move_to_opp_reward`、`push_opp_reward`。

## 8. 输出目录

推荐目录结构：

```text
reports/league/<run_id>/
  agents.yml
  tournament_config.yml
  fixtures.csv
  matches/
    <fixture_id>.json
  eval_cfgs/
    <fixture_id>.yaml
  leaderboard.csv
  match_matrix.json
  match_matrix.csv
  videos/
    <video_id>.mp4
  report.html
```

`fixtures.csv` 展示完整赛程：fixture id、A/B、slot assignment、checkpoint、seed、episodes、status、result summary、JSON 链接、视频链接。

`match_matrix` 展示 8x8 对战矩阵。矩阵单元建议显示 A 对 B 的聚合胜率、平局率、平均 reward margin，例如：

```text
win 0.62 | draw 0.18 | reward_margin +134.5
```

## 9. HTML 报告

最终 `report.html` 应包含：

1. 实验摘要：环境、日期、episodes、seed、候选数量、fixture 数、总局数。
2. 参赛者来源表：8 个智能体的 `agent_id`、训练线、run dir、checkpoint、source role、是否启用形态动作。
3. 赛制说明：双回合、side-swap、计分规则、形态开关规则。
4. 积分榜：总分、胜率、平局率、平均 reward、reward margin、左右边胜率、slot 胜率。
5. 对战矩阵：8x8 聚合矩阵。
6. 完整赛程表：56 个 fixture 的状态和链接。
7. 关键对局详情：冠军 vs 亚军、最佳 morph vs 最佳 fixed、异常高平局 pair。
8. 视频区域：嵌入或链接少量 mp4。
9. 可复现实验信息：命令、git commit、conda env、脚本参数、生成的 eval cfg。

## 10. 视频要求

不需要保存所有 56 个 fixture 的视频。建议保存少量定型视频：

- 默认保存冠军 vs 亚军的 1 到 3 局；
- 额外保存最佳 morph vs 最佳 fixed 的 1 到 3 局；
- 若用户指定 `--video_pair A B`，优先保存该 pair；
- 每个视频使用 mean action；
- 文件名包含参赛者、slot、checkpoint 和 seed，例如：

```text
videos/champion_vs_runnerup__A_agent0__B_agent1__seed1234_ep001.mp4
```

现有 `display.py` 可用于人工实时监控，但它使用 `render_mode="human"`，不能持久化视频。若要在 HTML 中放 mp4，需要新增或扩展 `rgb_array` 录制入口：

- 加载方式复用 `scripts/evaluate_checkpoint.py` 的 per-agent checkpoint 逻辑；
- 环境使用 `render_mode="rgb_array"`；
- 每 step 抓一帧；
- 用 `imageio` 写 mp4；
- HTML 报告中用 `<video controls>` 或链接展示。

## 11. 实现要求汇总

积分赛脚本至少支持以下输入：

- `--agents agents.yml`：8 个参赛者来源清单；
- `--base_cfg config/repro/unified-devant-training.yaml`：基础评测配置；
- `--episodes 50/100`：每个 fixture 的局数；
- `--seed`：基础随机种子；
- `--out_dir reports/league/<run_id>`：输出目录；
- `--video_pair`：可选，指定要持久化视频的 pair；
- `--video_episodes`：视频保存局数；
- `--smoke`：小局数快速验证模式。

必须产出的指标：

- 双方胜率、平局率、负率；
- episode 级积分和总积分榜；
- 平均 reward、reward 标准差、reward margin；
- episode length；
- agent0/agent1 slot 拆分表现；
- left/right side 拆分表现；
- 每个参赛者来源和 checkpoint 文件；
- 每个 fixture 的逐局结果。

必须产出的文件：

- `report.html`：主报告；
- `agents.yml`：参赛者来源；
- `fixtures.csv`：完整赛程；
- `leaderboard.csv`：积分榜；
- `match_matrix.json` / `match_matrix.csv`：对战矩阵；
- `matches/*.json`：逐 fixture 原始结果；
- `eval_cfgs/*.yaml`：每场临时评测配置，明确 `morph_optim_agents`；
- `videos/*.mp4`：至少一个指定 pair 或代表性 pair 的持久化视频。

核心原则：排名以胜负积分为主，reward 作为辅助解释；每个 pair 必须换 slot；每场必须按参赛者身份动态设置形态开关；报告必须能追溯每个智能体从哪个 run、哪个 agent、哪个 checkpoint 来。
