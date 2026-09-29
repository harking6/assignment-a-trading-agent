# Agent 经验学习层设计（memory + K线特征 + 舆论 + 知识库）

> 让多标 agent 越用越聪明：决策时存 K 线形态快照与舆论/知识上下文，下次按相似形态召回历史「决策→结果」对，喂回 LLM。单标图行为不动。

## 1. 现状缺口

| 组件 | 单标图 | 多标图 |
|---|---|---|
| memory | ✅ jsonl+chat_history | ❌ 完全没接 |
| K线特征 | ✅ technical_context 7指标 | ✅ 算了但**不进 memory**（用完即弃） |
| 舆论 | ✅ SentimentFeed（合成） | ❌ 写死 `sentiment_score=0.0` |
| 知识库 RAG | ✅ TradingRAG 注入 | ❌ 没接 |

多标 LLM 阶段实际上「无记忆、无舆论、无知识库、K线用完即弃」——每次再平衡都像第一次见市场，无法越来越聪明。

## 2. 目标：经验学习闭环

```
决策时:  存 K线形态快照 + 舆论 + 选中行业 + LLM reasoning  → memory
召回时:  按当前形态相似度 + 同市场观  → 召回历史「决策→结果」对
学习时:  回填上次决策实际超额  → LLM 看到「上次这种形态选电子赚/亏多少」
预测时:  RAG 召回行业知识 + 舆论情绪  → 增强本轮 prompt
```

四组件各司其职：memory 存经验、K线特征做相似度锚、舆论给方向性情绪、知识库给结构性认知。

## 3. 组件设计

### 3.1 memory + K 线特征快照

存储 schema（在 [memory层设计](memory层设计.md) 的 decision 记录上扩展）：

```json
{
  "type": "decision", "scope": "multi", "day": 14, "date": "2024-09-23",
  "market_view": "bullish", "benchmark_return": 0.0536,
  "selected_sectors": ["电子","电力设备"],
  "target_positions": {"000938": 700},
  "llm_reasoning": "...",
  "llm_trace": {"model_used":"deepseek-v4-flash"},
  "kline_features": {                          ← 新增：决策时形态快照
    "market": {"avg_rsi": 52, "avg_volatility": 0.18, "breadth_up": 0.42, "avg_dist_sma20": -0.03},
    "selected": {"000938": {"rsi":48, "volatility_20":0.15, "dist_sma20":0.02, "return_5d":0.05}}
  },
  "sentiment": {"market_score": 0.12, "top_events": ["电子板块资金流入"]},  ← 新增
  "active_return": null                         ← 回填
}
```

K 线特征分两层：**市场层**（全市场 rsi/波动率/涨跌家数/偏离均线）用于形态相似度，**个股层**（选中股 rsi/vol/dist_sma20/return_5d）用于复盘。

### 3.2 召回：相似形态 + 同市场观

```python
class TradingMemory:
    def recall_similar(self, kline_features, market_view, limit=3) -> List[Dict]:
        """按市场层形态相似度 + 同市场观 召回历史决策。

        相似度 = 1/(1+euclidean(market feature vector))，
        再用 market_view 同 regime 加权（避免下跌日召回上涨经验）。
        只召回有 active_return 的记录（已回填结果的）——没结果的没学习价值。
        """
    def to_lesson(self, records) -> str:
        """召回 → 「上次相似形态(bullish, RSI52)选电子→+2.1%；选银行→-0.8%」文本"""
```

形态向量 = `[avg_rsi, avg_volatility, breadth_up, avg_dist_sma20]` 归一化后欧氏距离。

### 3.3 舆论接入多标

`SentimentFeed` 现有合成实现（从 return/volume_z 推 score + events），扩展为多标版：

```python
class MultiSentimentFeed:
    def snapshot(self, universe, day, technical_context) -> Dict:
        """对 universe 每只股 + 全市场层面 给 sentiment score + events。

        - 个股层：复用 SentimentFeed 逻辑（return/volume_z 推事件）
        - 市场层：breadth(涨跌家数比) + 板块情绪（选中行业 avg score）
        - 扩展点: akshare stock_news_em / 财联社（可能被 block，合成兜底）
        """
        return {"market_score":..., "per_stock":{sym:score}, "top_events":[...]}
```

注入 `_stock_selector_node` / `_sector_selector_node` 的 LLM prompt（替代写死的 0.0）。

### 3.4 知识库 RAG 接入多标

`TradingRAG` 现有 lexical retriever，多标场景召回**行业知识/历史案例**：

```python
class MultiKnowledgeBase:
    def recall(self, sectors, market_view, k=3) -> Dict:
        """召回与选中行业 + 市场观相关的知识文档。

        文档来源（离线内置 + 可扩展）：
        - 行业基础：申万一级行业特征（电子=周期成长/银行=防御红利/...）
        - 历史案例：「上次 bullish + 电子RS领先 后 5 日表现」聚合统计
        - 规则手册：防御过滤/仓位规则
        检索：lexical（LC-04 已有），可换 Chroma/FAISS
        """
        return {"context":..., "sources":[...]}
```

历史案例可从 memory 聚合生成（memory 既是经验库也是 RAG 文档源——闭环）。

## 4. 数据流（多标图节点）

```
_universe_node
  ├─ recall_similar(当前kline_features, market_view) → state.memory_messages  ← 经验召回
  ├─ MultiSentimentFeed.snapshot() → state.sentiment                          ← 舆论
  └─ MultiKnowledgeBase.recall(预选行业) → state.rag_context                   ← 知识
_sector_selector_node
  └─ LLM prompt = 行业RS排名 + 上次相似经验 + 舆论 + 行业知识
_stock_selector_node
  └─ LLM prompt = 个股K线特征 + sentiment + 历史相似形态结果
_allocator_node
  └─ append_multi(决策 + kline_features + sentiment + reasoning)             ← 写经验
下一再平衡:
  └─ record_outcome(上次决策, 区间超额) → 回填 active_return
```

## 5. 越来越聪明的机制

| 机制 | 怎么变聪明 |
|---|---|
| 相似形态召回 | 见过这种 K 线形态 + 市场观，上次选啥赚/亏多少 → LLM 模仿成功避开失败 |
| 结果回填 | 决策→结果对积累，召回只取有结果的记录（学完的） |
| 舆论方向性 | 不再盲选，情绪极端时调整（市场恐慌时偏防御） |
| 知识库案例 | 「电子RS领先历史上 5 日胜率 60%」给 LLM 先验 |
| memory ↔ RAG | memory 聚合成 RAG 文档，经验越多知识库越丰富 |

随回测天数增加，memory 记录增多，相似召回命中率和质量提升 → LLM 决策越来越有依据。

## 6. 降级容错

- 任何组件失败不阻塞回测：recall_similar 异常→[]，sentiment 失败→0.0，RAG 失败→空 context。
- LLM 阶段仍有规则 fallback（既有降级链不变）。
- 合成舆论/lex RAG 保证离线可跑，真新闻/向量库是增强扩展点。

## 7. 扩展点（可选增强）

- 真新闻源：akshare `stock_news_em`（eastmoney，可能被 block）/ 财联社 / 雪球热帖；LLM 对新闻做情感打分。
- 向量库：Chroma/FAISS 替换 lexical retriever，语义召回。
- 基本面：财报数据（tushare fina_indicator，需 token）进 RAG。
- 形态识别：K 线形态（头肩底/突破）特征化进 kline_features。

## 8. 验证

- 单元：recall_similar 相似度排序正确；record_outcome 回填；to_lesson 文本含结果。
- 回测对照：924 段多标 LLM (a)无经验 (b)有经验召回，看超额/夏普是否随天数提升、重复错误是否减少。
- 单标 LC-03/LC-04 测试仍绿（单标路径不动）。
