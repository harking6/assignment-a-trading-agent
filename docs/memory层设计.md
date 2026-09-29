# 多标 Agent Memory 层设计

> 目标：让多标 agent 跨再平衡「记住」上次决策与结果，给 LLM 阶段提供经验上下文，形成 决策→结果→召回 的反馈闭环。单标图 memory 行为不动（LC-03 测试绿）。

## 1. 现状

`TradingMemory`（[agents/memory.py](../agents/memory.py)）：
- append-only jsonl（`data/agent_memory.jsonl`），按 `symbol` 过滤最近 N 条，转 `InMemoryChatMessageHistory`（human=situation / ai=decision）。
- **只在单标图 `run()` 用**（[graph.py:108](../agents/graph.py#L108)）：召回 → `memory_messages` state → LLM prompt。
- 多标图 `run_multi` 完全没接：不写、不召回。
- 无结果回填：记了决策，没记「上次选的股后来涨跌」。
- 无结构化召回：只能按 symbol 线性取最近 N 条，不能按市场观/行业/表现召回。

## 2. 目标

| 能力 | 说明 |
|---|---|
| 多标决策记忆 | 每次再平衡写 `scope=multi` 决策（市场观/选中行业/目标持仓/LLM reasoning） |
| 结果回填 | 下次再平衡回填上次决策的实际超额（决策→结果对，供 LLM 学习） |
| 召回注入图 | 多标图 universe 节点召回最近 N 次 multi 决策 → state → LLM prompt |
| 相似召回 | 按当前市场观召回同 regime 历史（避免下跌日召回上涨期经验） |
| 单标兼容 | 单标 memory 行为不变，jsonl 同文件共存（scope 区分） |

## 3. 存储 schema

jsonl 升级为带 `type/scope` 的结构化记录（向后兼容旧记录，缺字段补 None）：

```json
{
  "type": "decision",
  "scope": "multi",
  "day": 14,
  "date": "2024-09-23",
  "market_view": "bullish",
  "benchmark_return": 0.0536,
  "selected_sectors": ["电子", "电力设备"],
  "target_positions": {"000938": 700, "600584": 100},
  "llm_reasoning": "deepseek: 电子RS领先…",
  "llm_trace": {"model_used": "deepseek-v4-flash", "runtime": "langchain-openai"},
  "active_return": null
}
```

- `type=decision` 写于 allocator 节点后；`active_return` 字段初始 null。
- 下次再平衡（或回测结束）回填 `type=outcome`：`{"type":"outcome","ref_day":14,"day":19,"active_return":0.021}`，或直接 patch 原记录的 `active_return`。

## 4. TradingMemory 扩展方法

```python
class TradingMemory:
    def append_multi(self, day, date, sector_view, target_positions, llm_trace) -> None
    def recall_multi(self, limit=3) -> List[Dict]              # 最近 N 次多标决策
    def recall_by_regime(self, market_view, limit=3) -> List[Dict]  # 同市场观历史
    def record_outcome(self, ref_day, active_return) -> None  # 回填上次决策结果
    def to_context(self, records) -> str                       # 召回 → 文本注入 LLM prompt
    # 既有 recent(symbol)/chat_history(symbol) 不动
```

`to_context` 产出形如：
```
[上次决策 day14 bullish] 选中 电子/电力设备 → 结果 +2.1%
[上次决策 day9 neutral] 选中 银行/公用事业 → 结果 -0.8%
```

## 5. 与多标图集成（注入点）

```
universe → sector_selector → stock_selector → allocator
  ↑ recall_multi 注入            ↑ append_multi 写
  (state.memory_messages)        (记录本次决策)
```

- `_universe_node`：开头调 `recall_multi(limit=config.memory_window)` + `to_context` → 写 `state["memory_messages"]`。
- LLM 阶段（sector_selector/stock_selector）的 prompt 追加「最近决策与结果」段。
- `_allocator_node`：末尾调 `append_multi(...)` 写本次决策。
- 下一再平衡日 universe 节点召回时，`record_outcome` 回填上次决策的 `active_return`（用 engine 算的上次→本次区间超额）。

## 6. 反馈闭环

```
day N:   allocator 写决策 D_N
day N+k: universe 召回 D_N → 回填 D_N.active_return(用 N→N+k 超额) → 注入 LLM prompt
         LLM 看到「上次 bullish 选电子赚了 +2.1%」→ 本轮决策受经验调节
         allocator 写 D_{N+k}
```

## 7. 单标兼容

- jsonl 同文件，`scope` 字段区分 `single`/`multi`。
- `recent(symbol)` 仍按 symbol 过滤（单标记录有 symbol）。
- 多标记录无 symbol（portfolio 级），走 `recall_multi`。
- LC-03 测试（jsonl→chat_history）不受影响。

## 8. 降级与容错

- memory 读/写失败不影响回测：`recall_multi` 异常 → 返回 []，`append_multi` 异常 → 记 `last_error` 跳过。
- LLM 阶段即使有 memory 上下文仍可降级规则（memory 是增强不是依赖）。

## 9. 验证

- 新增 `test_multi_memory`：append_multi → recall_multi → to_context 内容正确；record_outcome 回填 active_return。
- 回测：924 段多标 LLM + memory vs 无 memory，看召回经验是否改善超额/减少重复错误。
- 单标 LC-03 测试仍绿。
