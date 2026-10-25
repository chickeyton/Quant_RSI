# 本地数据、独立推理与基线观察

本路径将本地输入接入已有 C2 Experiment Kernel，再封存实际入选模型，独立记录预测及可选的指示性账户。用途固定为 `developer_visible_observation`；没有 broker、后台行情服务、生产晋升或盈利承诺。工程 fixture 与公开 provider 文件的兼容性检查不构成用户市场 pilot 或长期 forward 验证。

## 1. 明确输入与知识时间

复制 [数据合同模板](../examples/local_data_contract.json.template)，补齐占位符。合同固定 provider 声明、稳定 source/instrument 身份、币种、raw 价格、带时区的逐 session 开收盘时间、成员资格来源和公司行动覆盖声明。时间偏移必须与注册的 IANA timezone 一致。当前导入器读取 UTF-8 事件 CSV；宽表 OHLCV 必须先显式转换，不能从日期猜测交易所时间或发布时间。

`columns` 将规范字段映射到实际 CSV 列名；表头必须恰好匹配映射。支持的字段为：

| 字段 | 要求 |
| --- | --- |
| `event_id`, `revision` | 稳定事件 ID、显式非负整数修订号；相同版本不同内容拒绝 |
| `kind`, `instrument`, `event_at`, `quality` | 必填；aware 时间，质量为 `valid`、`missing`、`invalid` |
| `session_id`, `price` | 报价必须指向注册 session/phase；有效价格为有限正十进制字符串，缺失/无效报价不带价格 |
| `source_available_at` | 来源可用时间；documented 模式每条修订必填 |
| `status` | 成员事件使用 `active`、`outside_universe`、`suspended` 或 `delisted` |
| `action` | 公司行动通知的明确描述；当前不执行拆股、分红或退市账务 |

外部事件仅允许 `OPEN_QUOTE`、`CLOSE_QUOTE`、`UNIVERSE_STATUS`、`CORPORATE_ACTION_NOTICE`。CSV 不能输入订单、成交、现金、时钟或检查点。完整注册 instrument 网格始终保留，每项带 eligible/ineligible/unresolved 状态及原因；退出 universe 不会删除真实持仓。每批最多64 MiB。

三种 `knowledge_basis` 必须明确选择：

- `documented_historical_availability`：提供冻结来源证据的 SHA-256、逐修订 source time；级别为来源声明，系统不替来源作独立认证。
- `assumed_historical_availability`：显式冻结 `available_at_event_time` 假设，结果只能用于历史探索。
- `observed_receipts`：按实际接收回执及来源时间共同限制可见性，不借用历史假设。

控制器在固定 registry 锁内记录 `first_seen_at` 和递增 `receipt_seq`。同一 source 下重命名文件、重复导入或换 request 不会改写原事件的首次接收时间；修订追加保存。新请求时钟倒退会被拒绝。source 合同不可在原身份下替换；代码不自动补全 PIT、公司行动、费用或货币元数据。

```bash
export QUANT_RSI_WORKSPACE="$PWD"  # 必须是已存在的绝对工作区
quant-rsi data ingest contract.json events.csv --request-id import-001 --evidence-file availability.bin
quant-rsi data status SNAPSHOT_SHA256
quant-rsi data replay SNAPSHOT_SHA256
```

只有需要 availability evidence 的合同才需要 `--evidence-file`，已保存同哈希证据可复用。历史文件今天导入，实际 first_seen 仍是今天。

## 2. 同一 C2 与实际模型封存

填写 [本地研究模板](../examples/local_research.json.template)。`data.source` 为 `local_event_snapshot`，引用已导入 snapshot；`sessions` 选择注册日历的前 N 项，`train_cutoff` 为从零计数的索引，`decision_delay_seconds` 显式描述历史收盘信息到决策的延迟，且必须早于下一开盘。当前 LocalDataSpec 继承24–1000 sessions、1–20 instruments 的 C2 范围；DataContract 更大的登记上限不扩大当前训练能力。`seed` 只固定 LightGBM 随机性，不生成或替换本地行情。

研究沿用同一 request、预算预留、worker、observation 与 incumbent 链。每个候选使用固定 LightGBM 和已有三特征语法；特征按各自 as-of 时间构造，训练只接收截止时已成熟且可用的 `open[t+2]/open[t+1]-1` 标签。缺失和不支持的公司行动不变成零标签。至少需要12条有效训练样本。

```bash
quant-rsi research run local-research.json --output runs/local-research --steps 1
quant-rsi research resume runs/local-research --steps 1
quant-rsi research replay runs/local-research
quant-rsi baseline seal runs/local-research
quant-rsi baseline replay BUNDLE_SHA256
```

封存必须来自已提交且可行的实际 incumbent，不能传入任意模型文本。bundle 固定预处理、输入合同、源检查点、训练信息截止、训练视图 as-of、真实 fit 完成时间和实际登记生效时间。今天导入或训练旧数据，不会刷新旧市场信息年龄。历史开发账务使用下述同一有限账户核心，但提交时钟为声明的历史场景，明确不算实际 forward 订单。

## 3. 独立推理与注册观察

`infer` 只加载固定模型，不需要未来 t+1/t+2 开盘价，不调用 fit 或 proposer。`--as-of` 和 `--basis` 可作历史分析；请求时间早于模型生效时间时，结果标记 `post_fit_historical_exploration`，不能作为当时已存在模型的证明。

```bash
quant-rsi infer BUNDLE_SHA256 SNAPSHOT_SHA256 SESSION_ID
quant-rsi infer BUNDLE_SHA256 SNAPSHOT_SHA256 SESSION_ID --as-of AWARE_TIMESTAMP --basis observed_receipts
```

选择 [纯预测模板](../examples/local_prediction_observation.json.template) 或 [账户模板](../examples/local_account_observation.json.template)。两者都必须填写 model/contract 引用、信息年龄与模型年龄上限和 aware expiry。纯预测模式没有本金、费用或结算参数。账户模式还必须显式填写全部财务规则。

```bash
quant-rsi observer register observation.json --episode-id baseline-v1
# 导入下一批真实收到的事件后：
quant-rsi observer run baseline-v1 --request-id poll-001 --steps 8
quant-rsi observer resume baseline-v1 --request-id poll-001
quant-rsi observer status baseline-v1
quant-rsi observer replay baseline-v1
```

注册先持久化唯一账户、模型和初始 receipt watermark，再记录提交完成后的 activation 时间。只有激活后发生且实际新收到的事件能进入 forward 观察；旧 bar、补录和重命名不能获得 forward 资格。观察器没有可传入的 fixture clock 或历史 now 参数。

每次 run 固定绝对步数上限，重复同 request 不追加额度；resume 不接受新的步数。没有待处理输入时返回 `awaiting_input`，后续 poll 使用新 request。一步可消费新回执，并推进至多一个市场 phase 和一项待记录预测。交易提交确认属于该步必须完成的后续事务，不额外增加额度。程序不等待下一天，也不在步数结束时平仓。

## 4. 有限账户规则与故障语义

首版只支持单币种、long-only top-k、正整数 lot、立即现金结算和库存解锁的模拟规则。买卖 commission、tax、slippage 分别显式输入，零也必须明填。滑点作用于 raw notional，commission/tax 按滑点后 notional 计算；notional 与各费用按注册币种精度逐项 half-even 舍入。最小收费、T+n 和复杂 venue 规则不受支持。

收盘形成原始数量；买单预留现金，不预支未成交卖款。全部保留持仓占用配置额度和仓位名额，拟卖但尚未成交的部分不能提前释放额度。下一注册开盘按卖先买后、同侧 instrument 顺序处理；逐 instrument 固定 fill cap、lot、原始数量、库存和预留现金共同限制成交。零可成交单位只撤余量，不生成虚假成交。

及时性由真实预测完成时间和 **订单意图事务提交之后**取得的 trusted witness 决定。witness 必须严格早于目标开盘；它自身可稍后持久化。崩溃丢失未持久化的 witness 时，不从 pending-start 或文件时间倒推出及时性，prepared 订单保守取消。已有有效确认的 elapsed 责任保留。

receipt cursor 与账户市场 cursor 分开。缺失一个目标开盘时，可先记账另一已知腿，保留未知腿的预留现金与责任；后续回执和推理继续，依赖该旧开盘的账户交易等待。迟到报价只解决原来的及时 standing order；修订不重定价已提交成交。后续修订或公司行动只追加可能影响旧结论的标记。

净值只使用当前收盘的新鲜持仓报价。缺失持仓价、未知成交或不支持的行动产生 unresolved/partial 结果，不丢仓、不补零、不拼接连续收益。现金基准为显式零利率。模型过期、市场信息过旧及风险暂停阻止新风险。

```bash
quant-rsi observer stop baseline-v1 --request-id stop-001
quant-rsi observer run baseline-v1 --request-id resolve-001 --steps 8
```

stop 是立即可持久化的控制意图，即使旧开盘缺失也能执行。`cancel_unelapsed_targets` 只撤停止回执之后的目标；等于或早于 stop 的未知责任继续等待。`preserve_inventory` 保留持仓；`queue_exit_at_next_eligible_open` 只能另行及时提交未来合法开盘的退出单，并按真实后续报价与费用结算。报告结束和 close 意图都不会凭空清仓。

## 5. 存储、恢复与验证

固定工作区的 `.quant_rsi/market/` 保存不可变 artifacts、数据身份与 `data.sqlite`。观察器使用同一数据库的独立身份及 episode/event 表。检查点 CAS、派生输出引用、消费游标和回执绑定在同一事务中提交；预提交孤立 artifact 不构成已提交结果。只读命令不迁移、修复、生成回执或写入 artifact。初始化身份存在但账本缺失时拒绝重新注资。

runtime 固定源码与依赖版本。修改源码后的新开发使用新身份；旧运行由相应 source snapshot 重放，不能修改 manifest 来绕过校验。AGENTS.md、已确认架构和既有 M0/C2/C4/C5 验收快照保持原样。

相关测试：`test_market_data.py`、`test_local_research.py`、`test_execution_account.py`、`test_observer.py`。测试覆盖独立现金 oracle、因果掩码、无未来开盘推理、旧信息年龄、双游标、提交边界、原子回滚、响应丢失、stop 和不重复注资。生产市场规则、实际 provider/PIT 选择、科学校准与长期 forward 观察仍需真实输入与独立验收。

## 6. 本地 C4 开发比较

用本节前述本地数据计划运行 C2，取得实际已提交观察后，沿用已有 C4 入口：

```bash
quant-rsi package run --evidence runs/local-research --output runs/local-c4 --state-policy reset --steps 1
quant-rsi package resume runs/local-c4 --steps 100
quant-rsi package replay runs/local-c4
# 只有 candidate 存在已提交且合格的 incumbent 时才封存：
quant-rsi baseline seal runs/local-c4/arms/candidate
quant-rsi infer BUNDLE_REF SNAPSHOT_REF SESSION_ID
```

源 C2 必须属于当前实现的基包，且没有待提交观察。原有提案规则需要同一 incumbent 下的真实窗口/算子配对支持；步数不足或算子没有严格胜过配对窗口时，`no_patch` 是有效结果，不产生父子研究。增加预算需预先写入新的研究计划，不能改写已注册运行。

一对父子各持有不可回收的完整 grant，使用同一封存 `data_ref`、知识口径、原始 receipt/first_seen、账户规则、M0 约束和 reset 初始因子。后续不同因子可以产生不同训练 mask、权重及结果。源 CSV、原 C2 目录在完整证据复制后不可用，或 live registry 后续新增修订，都不会改变已封存输入；不完整复制缺失的原始字节不能自动重建。推理模型发布仍需要可用且匹配的目标 registry。

只能经 `package resume` 推进受管理的子臂。授权核对实际外层账本、已提交 grant、路径/ID/plan/package 和当前轮次剩余额度；普通 context 字典、错误锁描述符以及向 C4 提交的 protected bound local 数据均拒绝。终态恢复不追加轮次或重跑 fit；失败保留已消耗额度和缺失分数。`status/replay` 不训练、不提案、不发布 artifact。

`tests/test_local_packages.py` 覆盖这些边界。其 CSV 明确由合成价格生成，仅用于工程路径覆盖；不代表真实市场数据或科学改进证据。本地 C5 通过独立 authority 授权接入派生开发视图与固定模型评分，见[操作说明](local_c5.md)。统计校准、生产晋升和多代驱动尚未实现。
