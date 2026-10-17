# 金融 LLM 结构有效性审计

本项目把金融 LLM 的可靠性从“输出看起来合理”变成了可运行、可检查的回测约束。它借鉴 Five Sins 框架，但不将合成数据测试包装为真实市场的偏差证明。

```text
交易决策
 ├─ 前瞻偏差：经验 recall 只取 outcome_day < decision_day
 ├─ 幸存者偏差：检查股票池快照 as_of；缺失时明确 Warning / N.A.
 ├─ 叙事偏差：每一个目标持仓必须在本次带因子证据的候选集内
 ├─ 目标偏差：空仓 / no-trade 是合法、可统计的决策
 └─ 成本偏差：总手续费、换手、LLM 调用次数与推理时延进入同一报告
```

## 回测展示

`TradingBacktestEngine.report()["bias_audit"]` 给出五项检查的状态、证据和说明；`to_dict()["bias_audit"]` 给出最近一次调仓的明细。这些字段不会修改既有 `sector_view` 合约，便于前端直接增设“结构有效性”卡片。

| 维度 | 可执行检查 | 失败/限制时的表达 |
| --- | --- | --- |
| 前瞻 | 经验结果日期严格早于决策日 | 记录被拒绝的未来 outcome 数，LLM 不可见 |
| 幸存者 | 动态股票池快照必须有 `as_of`，并与回测窗口核对 | 静态/合成股票池为 `not_applicable`，不虚报通过 |
| 叙事 | 目标股票是本轮因子候选集成员 | 标出不受支持的目标代码 |
| 目标 | 统计 `no_trade_decisions` | `cash_ratio=1` 且空权重被视为明确弃权 |
| 成本 | 手续费、换手、模型调用和时延 | 全部作为净效用的报告组成部分 |

## 复现

```powershell
.\.venv\Scripts\python.exe -B scripts\evaluate_factor_walk_forward.py --days 504
```

该脚本跑 4 个确定性合成情景、每个 504 个交易日，并将每一组风险预算消融的结构审计写到 `data/factor_walk_forward_results.json` 的 `structural_validity`。它验证的是约束、审计和组合构建链路；其中幸存者偏差在合成静态股票池下会诚实显示为 `not_applicable`。真实历史评测须接入逐日、含退市标的的点时股票池。
