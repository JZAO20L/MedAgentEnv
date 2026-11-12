# Mock 数据（演示与测试）

用于在没有 MIMIC / 医院数据授权的情况下跑通全链路。

| 文件 | 内容 | 被谁使用 |
|---|---|---|
| `cases.jsonl` | 6 个跨科室病例（内科/外科/呼吸/儿科/内分泌/急诊），含主观、客观信息与标准答案 | `src/medagent/env.py`、`scripts/cli.py`、数据集构造 |
| `knowledge.json` | 10 条医学 QA，覆盖 6 个病例的诊断与处置 | `MockKnowledgeBase`（替代 371k 的 ResponseMed.json） |
| `cost_reference.jsonl` | 18 条医疗服务/药品参考价 | `src/medagent/cost.py`、成本奖励 |

病例字段与真实数据集完全同构（`MedicalCase`），额外带两个非必需字段：

- `knowledge_keywords`：演示策略检索知识库时使用的关键词（不参与评分）
- `source: "mock"`：标记数据来源

替换为真实数据时，保持 `cases.jsonl` 的字段结构即可，不需要改环境代码。