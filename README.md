# Quant RSI

有限开发轨迹的入口、预算与恢复说明见 [Development trajectory](docs/development_trajectory.md)。

基于 [已确认架构](docs/architecture.md) 的量化研究系统。当前提供合成数据上的 **M0 固定模型内核、C2 动态研究循环、C4 受限研究包实验和 C5 工程保护评估**，以及 [本地数据、同 C2 研究、独立推理和持久化基线观察](docs/local_observer.md)。各运行内程序保持固定；新增 [五交易日价格预测基线](docs/forecast_baseline.md)；经校准的统计晋升和完整递归自改进仍未完成。

## 安装与运行

已在 Linux、Python 3.12 验证。进程控制依赖 POSIX `flock`、信号和资源限制；本版本不支持 Windows。安装依赖需要网络。合成实验无需外部服务；五日预测的自动采集需要访问公开行情和交易所日历，无需 API key。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .

quant-rsi run examples/development.json --output runs/demo
quant-rsi status runs/demo
quant-rsi replay runs/demo
quant-rsi resume runs/demo
```

`run` 使用新输出目录；已完成初始化的实验必须使用 `resume`，不能由 `run` 覆盖。示例包含 `baseline_return_1`、`momentum_5` 两个有效候选，以及应被拒绝的 `invalid_future_shift`。正常结果为两个 `completed`、一个 `rejected`；拒绝本身属于预期验证结果。

命令输出 JSON。`summary.json` 保存运行或恢复时的摘要；`status` 从账本重新读取当前状态。CLI 成功处理请求时返回 0，包括候选失败或预算耗尽；调用方应检查 `execution_status` 和每个候选的 `status`。配置错误、校验失败或运行锁冲突返回非零。

## 五交易日价格预测

~~~bash
export QUANT_RSI_WORKSPACE="$PWD"
quant-rsi forecast run SOXL --request-id soxl-first --output runs/soxl-first
quant-rsi forecast status soxl-first
quant-rsi forecast replay soxl-first
~~~

一条命令完成日线采集、来源保存、固定模型研究、封存和持久预测。同一请求 ID 与输出目录重用原始数据和结果；新一期预测使用新的 ID 和目录。支持范围、数据假设、输出解释及故障恢复见 [预测基线说明](docs/forecast_baseline.md)。

## 合成内核的计算约定

- 数据是固定随机种子生成的完整价格面板，使用工作日和合成 UTC 时间，不代表交易所日历、公司行动或真实可交易标的。
- `train_cutoff` 为从零计数的训练截止日。收盘 `t` 的标签是 `open[t+2] / open[t+1] - 1`；只有在截止日已成熟的标签可以训练。LightGBM 参数固定，每个候选拟合一次，开发期不重新拟合。
- 受限因子只支持 `momentum` 和 `mean_return`，窗口 1–30。特征包含该因子、单日收益率和当日开收盘收益率；不执行候选提供的 Python 代码。
- 收盘 `t` 形成整数股目标和订单，开盘 `t+1` 成交。订单只使用当时已知的现金、持仓、收盘价与预测；未来开盘跳空可以影响实际成交数量。
- 组合为等权目标的 long-only top-k，先卖后买，同侧按标的顺序成交；现金不足时部分成交。费用和滑点均按单边基点计算。最后一个标签区间结束时强制平仓，并扣除退出费用和滑点。
- 主指标为相对零利率现金基准的**净主动收益算术均值 × annualization**，不是 CAGR。最大回撤按各区间结束的开盘净值计算；超限结果保留，但不进入开发期最优候选选择。
- `development_incumbent` 仅是本次开发比较的最优可行候选。所有结果均标记 `development_only`、`capacity_status=unverified`、`promotion_decision=not_requested`。

## 实验账本与恢复

```text
runs/demo/
  manifest.json       # 计划、数据和运行环境的内容哈希引用
  journal.sqlite      # 候选状态、预算占用、事务性事件
  artifacts/<sha256>  # 数据、模型文本、预测、账务、日志等产物
  work/0000/          # worker 输入、原始日志和原子提交的输出
  summary.json        # 可读摘要
```

每个候选先占用一次尝试额度；无效因子也计数。有效候选启动前占用一个完整的 `worker_timeout_seconds` 额度，计入 `max_worker_seconds`，完成、失败和中断均不退款。这是保守的 worker 时间槽预算；实际经过时间另记为 `elapsed_seconds`，恢复时无法测得则为 `null`。

创建阶段先原子写入 `initialization.json`，固定计划和环境；账本建表、身份和 `run_created` 事件在同一 SQLite 事务中提交，之后移除初始化记录，才允许启动候选。创建中断时，可用 `resume` 或相同计划的 `run` 完成初始化；不同计划会被拒绝。若中断发生在首条记录发布前，只能用 `run` 重试空目录（可能含锁或临时文件）。已完成初始化但账本缺失、身份冲突或结构损坏的实验不会被自动重建。

worker 输入只有一份带完整哈希的计算绑定，计划和数据按内容哈希加载，因子必须与计划内该候选一致。worker 在计算前验证绑定，父进程在接受结果时核对落盘输入和返回身份，避免错误载荷沿用原提案的身份通过评分或重放。

单个实验只允许一个 controller；worker 继承运行锁，并有独立截止信号。父进程中断后，worker 尚未退出时恢复会被拒绝；稍后可重试 `resume`。存在完整且绑定正确的 worker 输出时，恢复只完成验证和记账；没有有效输出则记为 `interrupted`。终态候选不会隐式重跑。

`replay` 验证内容哈希、计算绑定及运行环境，从已保存的预测重新计算全部订单、成交和净账务，并要求与保存报告一致；它不会重新训练或产生新的证据。源码或已安装包版本改变时，`resume`/`replay` 会拒绝执行；恢复原环境后再运行。`status` 仍可读取旧结果。实验目录可以整体移动，但不要编辑内部文件。哈希用于发现损坏和不一致，不是对拥有文件写权限者的签名认证。

## 开发与验证

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

测试包含真实 LightGBM worker 的端到端运行、独立手算账务、未来数据扰动、完整预测覆盖、预算耗尽、无效输出、锁互斥、产物损坏和崩溃恢复。没有设置覆盖率百分比门槛。

目录与职责：

| 路径 | 内容 |
| --- | --- |
| `src/quant_rsi/schemas.py` | 严格 DevelopmentPlan 与受限因子定义 |
| `src/quant_rsi/data.py`, `model.py`, `worker.py` | 合成数据、因果特征、固定训练与子进程入口 |
| `src/quant_rsi/accounting.py` | controller 使用的独立账务与主指标计算 |
| `src/quant_rsi/controller.py`, `journal.py`, `artifacts.py` | 调度、生命周期、预算和产物校验 |
| `market_contracts.py`, `market_data.py`, `asof.py`, `local_research.py`（均在 `src/quant_rsi/`） | 显式本地合同、不可变回执、因果视图和同 C2 本地输入 |
| `baseline.py`, `execution_account.py`, `observer*.py`（均在 `src/quant_rsi/`） | 实际模型封存、独立推理、共享账务和有限步观察器 |
| `tests/` | pytest 测试，文件使用 `test_*.py` 命名 |
| `examples/development.json` | 可直接运行的开发计划 |
| `docs/`, `discussion_boards/` | 已确认设计、评审讨论和历史记录 |

Python 使用四空格缩进、`snake_case` 函数和模块、`PascalCase` 类；Ruff 负责格式与静态检查，行宽 100。依赖版本固定在 `requirements-dev.lock`；修改依赖时同步更新并验证。提交应使用简洁祈使句，PR 说明目的、行为变化和验证结果。详细运行命令以本 README 为准；原有 `AGENTS.md` 按要求保留。

## C2：有限步动态研究

```bash
quant-rsi research run examples/research_development.json --output runs/research_demo --steps 1
quant-rsi research status runs/research_demo
quant-rsi research resume runs/research_demo --steps 3
quant-rsi research replay runs/research_demo
```

每次 run/resume 的 --steps 为 1–100 的正整数，默认 1，表示本次调用最多新提交的 observation/checkpoint 数。未提交观察的 pending 步骤恢复完成后占其中一步；在上次调用已经提交的步骤不重复计数。调用在提交边界返回 paused/step_limit 后，不留后台 controller，不再提案或派发；只有显式 resume 才继续。此有限步流程就是本版本的停止方式，没有异步 stop/cancel 命令。预算或研究停止条件满足时返回 completed 和具体 stop_reason；终态 resume 不启动新研究。

进程中断后，status 中 running 表示有未收尾状态，并不保证进程仍活跃。用 research resume 处理原步骤；尚存活的 worker 通过继承锁和 deadline 阻止并发恢复。已提交提案和结果复用；未发布提案的纯确定性计算只允许按同一请求/父状态重算，不重新预留 attempt。无有效 worker 输出则保留真实 interrupted/failed 及原费用。旧步骤的恢复、继续新的合法步骤和只读 replay 是不同操作。

冻结的 neighbor_search_v1 从 baseline 开始，依次检查当前 incumbent 的同算子窗口 +1、-1、另一算子同窗口，跳过越界、训练样本不足或已见邻居。只有可行候选的主指标严格改善超过有限非负 min_improvement，才移动开发 incumbent 并将 no-progress 清零；相等不移动。baseline 有效时初始化 incumbent/no-progress=0，否则停止。之后失败、重复、不合格和不改善各递增一次 no-progress；达到配置上限、注册邻域耗尽或预算不足就停止。

所以同样试过 momentum(6)，改善时可继续 momentum(7)，未改善时可回到 momentum(4)。这是固定程序下的记忆/搜索状态学习，不是 researcher package 自改写。候选逐个形成新的封存 singleton DevelopmentPlan，原模板和旧计划不修改；模型、数据与会计仍走原 Kernel。仅允许已提交且当前 task 可见的 development observation 影响提案。重复开发选择的收益不能作为独立验证证据。

全 campaign 只有一个 SQLite 总账：proposer 调用前预留 1 attempt，baseline 也计数。attempts_reserved = kernel_attempts + unmapped_proposal_reservations；尚未物化候选、生成失败等预留不会消失。每次实际启动按完整 timeout 计 worker-slot，启动前崩溃、失败和中断也不退款。剩余完整 slot 不足时不派发。checkpoint 不含可覆盖额度，续跑不会重置预算。未接入 LLM，不报告虚构的 token/billing 成本。

程序代码/配置作为 ResearcherPackage 固定；MemorySnapshot 保留 observations，ResearchStateCheckpoint 保存父状态、incumbent、visited、no-progress、continuation 和 ledger 身份。请求、实际响应、决策、每个 experiment manifest 及 provenance 都是内容寻址产物；proposal response 的原文另保存在 proposals/。候选和 Kernel plan 内的 proposal_index 分别标识，避免单候选计划与全局 attempt ID 混用。

接受缓存与语义重放会按冻结的纯策略校验预期提案。合法但不符合该请求预期的缓存候选也会被明确拒绝，保留 response artifact 和 reservation，不训练替换候选。真实 proposer 失败则保持缺失分数。observation 应用、父 checkpoint CAS 和事件在同一事务提交，避免重复学习。

报告提供数据/package/checkpoint 引用、输入 evidence、假设与失败原因、相对固定 baseline/前一 incumbent 的开发差值、选择结果、剩余额度和停止原因。逐次 elapsed_seconds 为实测值，恢复时不确定则为 null；不要与逻辑 worker-slot 额度混淆。status/replay 不发布研究状态；replay 还重新计算 Kernel 账务和研究状态转移，不产生新请求。

新增模块：researcher.py 为冻结策略，research.py 为 campaign 编排/报告，research_journal.py 为同数据库内的 reservation/观察事务；对应回归在 tests/test_research.py。保留已验收版本的源码/依赖快照（M0 位于 runs/accepted_source/m0_f54aa55ddba7），以匹配源码锁定的旧运行；不要用新源码覆盖旧运行产物。

## C4：受限研究包实验

先用当前源码生成可重放的基包证据，再显式启动一次补丁实验：

```bash
quant-rsi research run examples/research_development.json --output runs/c4_evidence --steps 100
quant-rsi package run --evidence runs/c4_evidence --output runs/c4_trial --state-policy reset --steps 1
quant-rsi package status runs/c4_trial
quant-rsi package resume runs/c4_trial --steps 1
quant-rsi package resume runs/c4_trial --steps 100
quant-rsi package replay runs/c4_trial
```

一次 package campaign 只有一个 change request 和至多一对父子研究。ChangeProposer 在生成前永久预留一次请求。源 run 必须属于当前源码的 window_first 基包、没有 pending observation；入口核验完整已提交链与 Kernel 账务，然后封存只读证据副本。旧 C2 证据不绕过版本检查导入；可用原快照重放，或在新引擎下创建新的研究。

冻结规则按提交顺序寻找最近一组共享同一真实 incumbent 的合格观察：同算子相邻窗口与同窗口另一算子。换算子分数严格高于该组最后一个窗口候选时，生成 operator_first 补丁；否则返回明确 no_patch，包括证据不足。该规则只提出开发假说，历史证据重用不等于独立验证。实际响应单独保存；exact base、来源/水位、接口、reset、唯一字段和值，以及冻结程序的完整预期输出均受校验。非法或错误缓存保留原文、拒绝原因和已扣请求，不采样替代响应。

唯一干预是 policy.neighbor_order：window_first 顺序为窗口+1、-1、换算子；operator_first 顺序为换算子、窗口+1、-1。搜索域、模型、数据、费用、风险、指标和额度不变。candidate 包绑定 parent/patch/proposer/request 谱系，父子 run 显式绑定各自 package，从同一 baseline 和空 memory 重置启动。迁移、跨包 checkpoint、独立 research resume 子臂均拒绝；用 package resume 统一恢复。

本地 CSV 的 C2 证据也可通过相同 `package run` 入口使用。父子读取证据副本中同一份 `LocalResearchData` 与账户规则，不重新读取源 CSV 或最新 registry；启动校验实际 C4 grant、锁、包、输入及持久轮次额度。已有合格后代模型可通过 `baseline seal runs/c4_trial/arms/candidate` 封存，再使用 `infer`。没有满足真实配对条件时仍返回 `no_patch`。本地 C5 使用独立登记任务、裁切开发视图和原账户规则，见[本地 C5 工程比较](docs/local_c5.md)。详见[本地 C4 操作说明](docs/local_observer.md#6-本地-c4-开发比较)。

外层在任一子臂初始化前原子拨付两个完整 grant：总研究 attempt 和 worker-slot 上限各为单臂计划的两倍。grant 固定目录/ID/plan/package，永不回收；子臂只在原账本扣实际消耗。research_budget 分别报告限额、granted、consumed、outstanding 与 unallocated，始终满足：
- limit = granted + unallocated；
- granted = consumed + outstanding。
历史源证据消耗单列，patch request 单列；worker-slot 是保守逻辑额度，不是实测耗时。未启动或未知的 grant 不释放，失败不伪装成零收益。

--steps 为本次**每臂**最多新提交的观察数。每轮先封存目标计数；崩溃后用原 --steps 恢复未完成轮次，已提交观察不重复执行。若上一轮已完整提交，再次 resume 是新一轮有限继续。子臂初始化回执在执行前持久化，已初始化但损坏/丢失的账本不会重建。worker 同时继承外层和子臂锁，孤儿进程未退出时不能绕过锁恢复。

status/replay 仅观察，不提交缺失的外层回执或补造 artifact。committed_completion_ref 表示已提交的外层最终结果；子臂完成而此字段为 null 时，显式 resume 才收尾。终态 resume 不产生新请求、实验、轮次或事件。不存在 promote 命令；accepted_pointer 是本次只读工程基准引用，不是 C5 晋升注册表。候选开发成绩不能更新它。

C4 引擎生成新的基包身份，并标记相对已验收 C2 的工程来源，不把旧 digest 贴到新源码。原 C2 源码位于 runs/accepted_source/c2_16b85204c890/。新模块为 changes.py、package_journal.py、package_experiment.py；机制、负例、初始化/提交窗口和孤儿锁回归见 tests/test_packages.py。真实合成示例中，父子可分别完成6和8次实验、最终开发差值为0；这不是收益改善证明。


## C5：工程保护评估

参见 [C5 运行与恢复指南](docs/c5_protocol.md)及[本地 C5 输入与会计边界](docs/local_c5.md)。必须显式设置绝对路径 `QUANT_RSI_WORKSPACE`，其唯一 authority 统一记录证据曝光、开发 grant、评分 slot、报告和 registry。修改输出目录或标签不能重新获得额度。

```bash
export QUANT_RSI_WORKSPACE="$PWD"
quant-rsi research run examples/research_development.json --output runs/c5_evidence --steps 100
quant-rsi package run --evidence runs/c5_evidence --output runs/c5_packages --state-policy reset --steps 100
quant-rsi evidence init examples/evidence_program.json
quant-rsi compare register examples/comparison.json --output runs/c5_comparison
quant-rsi compare run runs/c5_comparison --steps 100
quant-rsi compare replay runs/c5_comparison
quant-rsi compare decide runs/c5_comparison --request-id first-retain
```

首版仅支持 `engineering_only`：真实执行开发搜索和固定模型评分，但强制留任；`confirmatory` 在任何评分数据读取或新 grant 之前拒绝。`initial` 是 `parent` 的同一运行/模型/评分别名，不另计预算或经济质量。每个 task 有80个开发日、40个评分决策日及2个成交/清仓尾日。

新源码需要新 runtime/package 身份。原 C4 r2 源码保留在 `runs/accepted_source/c4_b4ad09985d3e/`；历史运行使用对应快照重放，不修改旧 manifest。C5 的具体模块为 `evidence.py`、`comparison.py`、`protected_scoring.py` 和 `promotion.py`；故障与会计回归见 `tests/test_comparison.py`、`tests/test_protected_accounting.py`。

## 后续能力边界

本地适配器支持显式映射的事件 CSV，观察器支持纯预测或有限规则下的指示性账户；它们不等于完成真实市场 pilot 或 broker paper trading。当前没有在线 provider/broker、Qlib/RD-Agent 适配、LLM 提案、自改代码、经校准的确认性推断、生产晋升或 M1 定期重训。C5 的 CI/p/power 为 null 且附具体未校准原因，formal tests 为0。配置拒绝不支持的计划和模式。worker 是受信计算的进程边界，不是运行任意生成代码的安全沙箱。

后续应按 [架构中的能力依赖与验收条件](docs/architecture.md) 逐项扩展；合成数据的收益结果不能用来推断策略有效性。

## 本地数据与持久化观察

按 [本地观察指南](docs/local_observer.md) 填写合同和计划。市场、日历、provider、历史可用性、标的集合、币种和费用必须显式选择；`examples/local_*.json.template` 中的占位符会被验证器拒绝，不能直接当作市场默认值使用。

```bash
export QUANT_RSI_WORKSPACE="$PWD"
quant-rsi data ingest contract.json events.csv --request-id import-001 --evidence-file availability.bin
quant-rsi research run local-research.json --output runs/local-research --steps 1
quant-rsi baseline seal runs/local-research
quant-rsi infer BUNDLE_SHA256 SNAPSHOT_SHA256 SESSION_ID
quant-rsi observer register observation.json --episode-id baseline-v1
quant-rsi observer run baseline-v1 --request-id poll-001 --steps 8
quant-rsi observer replay baseline-v1
quant-rsi observer stop baseline-v1 --request-id stop-001
```

`data ingest` 和 `baseline seal` 返回后续计划所需的哈希引用。每次新行情导入后使用新的 poll request；中断恢复使用原 request。`status`/`replay` 不修复或写入状态，不调用训练或提案器。尚未选择本金、费用或结算规则时，可使用无财务字段的 `prediction_only` 计划；市场日历、标的身份、知识口径及完整 DataContract 仍须明确提供。
