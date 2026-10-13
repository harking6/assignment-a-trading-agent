# AIC 多智能体模拟交易系统

本项目是面向 A 股研究与模拟交易的多智能体系统。系统同时支持单标的研究决策和 Top-Down 多标的选股，通过分析师、研究辩论、风险审查、组合规划与模拟券商形成完整闭环。

项目当前已经实现 LangChain/LangGraph 工作流、离线决策模式、可选 OpenAI 兼容模型、确定性模拟行情、多数据源行情、动态股票池、股票池快照重放、模拟成交、基准比较、记忆回填以及浏览器工作台，不再是待补全的 starter。

## 核心链路

```text
浏览器 / HTTP API
  -> MarketEnv
  -> TradingBacktestEngine
  -> TradingAgentsGraph
     -> 单标：分析师 -> 研究辩论 -> 风险委员会 -> 组合经理
     -> 多标：市场状态 -> 行业 -> 个股 -> 辩论排序 -> 组合规划 -> 分配
  -> 再平衡订单规划
  -> SimulatedBroker
  -> 状态、报告、记忆与前端展示
```

架构边界、已实现接口和数据流见 [架构边界与数据流](docs/架构边界与数据流.md)，完整环境、测试、快照重放和团队协作步骤见 [复现与协作指南](docs/复现与协作指南.md)。

## 环境安装

要求 Python 3.11。以下命令在项目根目录的 PowerShell 中执行：

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
Copy-Item .env.example .env
```

`.env` 只用于本地密钥和模型配置，已被 Git 忽略，不得提交。

## 启动

```powershell
.\.venv\Scripts\python.exe app.py
```

浏览器访问 `http://127.0.0.1:8011/`。终端显示的静态资源和 `/api/state` 访问记录属于正常请求；按 `Ctrl+C` 停止服务时出现 `KeyboardInterrupt` 代表人工终止，不是业务故障。

默认可使用 `synthetic` 数据与 `offline` Agent 模式完成无密钥演示。需要在线 LLM 时，在本地 `.env` 中配置 OpenAI 兼容接口；LLM 只生成研究与交易意图，最终现金、持仓、费用、滑点和整手约束仍由 `SimulatedBroker` 执行。

## 数据源

- `synthetic`：确定性离线行情，适合开发和回归测试。
- `csv`：读取 `data/` 下的本地 OHLCV 文件。
- `akshare`：A 股历史行情、实时报价和动态股票池。
- `tushare`：A 股历史行情，需要本地 `TUSHARE_TOKEN`。
- `yfinance`：美股与 ETF 行情。

中国证券代码中，`000001` 表示平安银行；`000001.SH` 或 `sh000001` 表示上证指数。A 股历史价格默认采用前复权 `qfq`，实时报价为未复权价格。

## LLM 配置

本地 `.env` 示例：

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_DEEP_MODEL=deepseek-reasoner
TRADING_AGENT_MODE=auto
```

不要把真实密钥写入 README、源码、测试结果或参赛材料。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -B scripts\check_assignment.py --strict
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -v
```

架构模块的专项测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest tests.test_architecture_parity tests.test_architecture tests.test_engine_injection tests.test_universe_snapshot tests.test_rebalance -v
```

## 主要目录

```text
agents/                  LangGraph 工作流、角色 Agent、记忆与 Top-Down 选股
core/schemas.py          跨模块领域模型与回测配置
core/contracts.py        Graph、行情和股票池的运行时协议
data/providers.py        synthetic/CSV/AkShare/Tushare/yfinance 行情源
data/universe_service.py 可保存和重放的股票池快照
engine/backtest.py       回测编排、执行与报告
engine/rebalance.py      无副作用的目标持仓差分与订单意图规划
tools/broker.py          现金、持仓、费用、滑点和成交约束
tests/                   单元、契约、等价与依赖边界测试
app.py                   本地 HTTP 服务
index.html/app.js/styles.css 浏览器工作台
```

## 当前架构边界

- 成员 A：协议、依赖注入、可复现股票池、订单规划和架构测试。
- 成员 B：记忆样本门控与召回偏置。
- 成员 C：Prompt、JSON 容错、行业 RS、质量因子、回测与消融实验。
- 成员 D：`TradingGraphState`、Graph 编译、前端与展示材料。

架构层不会擅自调整策略阈值、因子权重、Prompt、记忆算法或 Graph 拓扑。团队修改和合并规则以 [复现与协作指南](docs/复现与协作指南.md) 为准。
