# 本地 C5 工程比较

本路径将已完成的本地 C4 包配对接到同一 C5 authority。它运行实际父子开发、封存选中模型、按登记日历评分并执行 retain；它不提供独立样本资格、统计晋升或真实市场盈利证据。公开历史及已用于 C2/C4 的来源保留原暴露身份。

## 1. 冻结输入

先按 [本地数据说明](local_observer.md) ingest 实际事件 CSV，并完成一次产生 `patch` 的本地 C4。没有真实配对改动时保留 `no_patch`，不能手写一个候选包代替。

设置绝对路径 `QUANT_RSI_WORKSPACE`。既有 market registry 可以与首次 C5 初始化共存；一个 workspace 只有一个不可重置的 C5 program。新名称、输出目录或输入路径不提供新额度。旧运行使用原源码快照。

本地 program JSON 的外层字段为 `name`、`source: "local_event_snapshot"`、`package_campaign` 和 `tasks`。每个 task 必须包含以下字段：

| 字段 | 要求 |
| --- | --- |
| `task_id` | 小写字母开头的字母、数字、下划线标识；不可重复 |
| `data` | 实际 `LocalDataSpec`，包括 ingest 返回的 `snapshot_ref`、完整 `sessions`、`instruments`、`train_cutoff`、`seed`、`decision_delay_seconds` |
| `development_sessions` | 登记日历中的开发前缀长度；至少 24，且满足原训练/开发最小样本约束 |
| `score_session_ids` | 开发前缀之后所有登记 session 的精确有序列表，至少 3 个；不补缺失日期 |
| `terminal_session_id` | 上述列表最后一个 session |
| `weight` | 正十进制字符串；所有 task 的权重精确合计为 `"1"` |
| `economic_group` | 显式登记的经济任务组 |
| `dependence_cluster` | 当前仅接受 `"unverified_common_cluster"` |
| `prior_exposure` | `"public_or_previously_exposed_engineering"` 或 `"unverified_engineering_only"` |

允许 1–4 个任务。训练配置、账户规则、研究策略与预算继承冻结 C4；只有新 snapshot、总 session 数及开发视图绑定改变。当前每角色预算固定为 8 次、每次 20 秒。资产、货币和实际时间相同的评分覆盖不能通过源名、session 标签、CSV 排序或等价时区换取重复权重。任一任务的开发前缀不得暴露另一任务的评分区间。

## 2. 运行与恢复

在填好 `local_program.json` 的真实引用和 session 列表后运行：

```bash
quant-rsi evidence init local_program.json
quant-rsi compare register examples/comparison.json --output runs/local_c5
quant-rsi compare run runs/local_c5 --steps 100
quant-rsi compare status runs/local_c5
quant-rsi compare replay runs/local_c5
quant-rsi compare decide runs/local_c5 --request-id retain-local-1
```

中断恢复使用 `compare resume`，并保持未完成调用原来的 `--steps`。每任务有父、子两个唯一开发/评分运行；initial 是 parent 的逻辑别名，不另领预算。首次两个评分共扣 40 秒，技术重试最多各一次、每次另扣 20 秒。UNKNOWN 不退回曝光或费用；模型损坏、缺价及语义失败不转为现金。现金仅来自登记的、权威确认的开发无有效 incumbent。

## 3. 数据与模型边界

注册器读取原始来源并记录单独的 registrar exposure。开发 worker 只得到新身份的裁切 CSV、snapshot 和派生 receipt，保留原 raw/receipt/payload hash 与 first_seen 谱系，并另记派生创建时间。原始可用性文件内容和保护 CSV 不进入开发 artifact。提前公布且当时已知的行动通知继续可见。

C5 开发授权核验实际 authority 锁、task、package、plan、view、grant 和当前有限操作；不能冒充 C4 授权。选择封存依据实际训练 mask、成熟标签、模型权重与开发预测。评分只推理，不训练或重新选模型。

OPENED 必须先于评分器读取原文。首次接受评分时，对照登记来源核验消费证据，并提交绑定 attempt、selection、task、result 的验证 receipt。只读 replay 从这些已封存证据验算，不重新打开保护原文，不修复账本或发布 artifact。

## 4. 收益网格与终止

每个评分从登记 NAV0、空持仓、空订单、无应计现金开始。首评分 session 收盘作出第一决策，最早下一真实 open 成交；该首日现金收益也计入固定 T。每个登记 close 必须有一条估值记录，缺失收益不缩小分母。

严格继承 `AccountRules.termination`：`preserve_inventory` 在末日以真实 fresh close 估值，保留持仓和未实现资产，明确退出费用未发生；`queue_exit_at_next_eligible_open` 在末日前一决策提交退出订单。尾日缺价、成交上限不足、未解决订单、公司行动或风险失败保留完整账户状态，要求退出时残留持仓会使评分失败。没有免费清仓或现金替代。

CI、p、power、统计 calibration/feasibility 继续为 null，并保留原因；formal tests 为 0。确认性模式在保护读取和发放 grant 前拒绝。accepted registry 仅允许带 package/revision/plan/report 绑定的 retain CAS；初始化 parent 引用不代表科学优越性。

针对性验证：`pytest tests/test_local_projection.py tests/test_local_comparison.py`。组件会计和 mock 事务测试与实际 CLI、wheel、进程中断证据分别记录。
