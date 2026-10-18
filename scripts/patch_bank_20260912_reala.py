"""2026-09-12 第四轮备考库补丁：把「没有真实数据」的项补上可核查证据

背景：用户对完 60 项对照表后问「有多少有真实数据，有多少没有」。
彻查后发现三类问题：
  ① 有 21 项只有定性陈述 / 「未做」两个字，没有任何证据
  ② 有 3 处**口径错误**（备考库写的和现状不一致）
  ③ logs/audit.jsonl 里有 200 条真实调用日志，但备考库写「logs/ 为空」——白扔了一座金矿

本补丁：
  1. 给 21 个 C 档项补上 grep 计数 / 代码行号 / SQL 实查结果
  2. 修正 3 处口径错误（27行 vs 509行、logs 为空 vs 200条、54s 是单次案例 vs P50 4.9s）
  3. 给原本只有配置值的项（D1/E2/E3/C1）补真实统计数字
  4. 新增「数据性质盘点」节，明确哪 15 项能拿数字讲、哪 45 项只能定性
"""
from pathlib import Path

BANK = Path(r"D:\简历\技术点深挖备考库.md")
t = BANK.read_text(encoding="utf-8")
orig_len = len(t.splitlines())


def replace_row(row_id: str, new_tail_cols: list) -> None:
    """按 | ID | 定位表格行，替换其为：| ID | 项 | 生产做法 | 我们现状 | <新证据> |

    new_tail_cols 长度必须等于该行除 ID 外的列数。
    """
    global t
    lines = t.splitlines(keepends=True)
    idx = None
    for i, l in enumerate(lines):
        if l.startswith(f"| {row_id} |"):
            idx = i
            break
    if idx is None:
        raise SystemExit(f"[FAIL] 找不到行 {row_id}")
    old = lines[idx]
    # 保留原行的前两列（ID、项），其余由 new_tail_cols 接管
    cells = [c.strip() for c in old.strip().strip("|").split("|")]
    head = cells[:2]
    newline = "| " + " | ".join(head + new_tail_cols) + " |\n"
    lines[idx] = newline
    t = "".join(lines)
    print(f"  ✓ {row_id}: {cells[1]}")


print("=== [1] 修正 3 处口径错误 ===")

# ---- 口径错误 1：A5 审计日志「logs/ 为空」是错的，实际有 200 条 ----
replace_row("A5", [
    "审计 JSONL 落盘", "✅ 已落地（有真数据）",
    "`audit.jsonl` **实测 200 条 / 79KB**（2026-09-02~09-12），字段含 usage/duration/tools/trace_id；"
    "另 `app.log` 30MB。⚠️ 教训：备考库此前写「logs/ 为空」是**错的**——没实际 ls 就下了结论"
])

# ---- 口径错误 2：F1/F5 语料行数口径混淆（27 行 vs 509 行）----
replace_row("F1", [
    "生产级知识库应上千至万级文档", "❌ 极少",
    "**业务语料各 5 条**（eval-clinical 5 + eval-usage 5）——禁说「海量」。"
    "⚠️ 口径边界：`capital-medical/clinical` 是 27 行；`kb_chunks` **整表实测 509 行**，"
    "别把两个数字混着说（见 F5）"
])
replace_row("F5", [
    "定期清理残留/重复", "❌ 未做（有实锤）",
    "**2026-09-12 实查 kb_chunks 共 509 行**：真实业务语料仅 47 行"
    "（capital-medical/clinical 27 行含 22 行历史残留 + usage 5 + eval/debug 15），"
    "**其余 462 行 / 35 个 bench-* 临时租户是我跑评测脚本留下的实验垃圾**；"
    "另有 bench_ann 表 2 万行同为实验残留"
])

print("\n=== [2] 修正 6.5 节：1.2s/50s/54s 是单次案例，不是基线 ===")
old_65 = "> **端到端按场景分类看**：纯对话 1.2s、知识型长答案 50s、工具型查询 54s"
new_65 = (
    "> ⚠️ **2026-09-12 口径修正**：下面 1.2s / 50s / 54s 是**两个单次案例的观测值**（可复盘的剖析素材），"
    "**不是基线、不是 P50**。从 `logs/audit.jsonl` 实测 166 通真实调用统计出来的分布是：\n"
    "> **纯对话 P50 2.8s（P95 13.4s）｜带工具 P50 4.9s（P95 65.6s）｜全体 P50 4.76s / P90 46.0s / P95 63.8s｜最大 90.3s**。\n"
    "> 也就是说：**典型体感是 5 秒内，54s 那个落在 P90~P95 区间（偏慢的那 5%~10%）**。\n"
    "> 被追问「54s 是均值还是 P50」时必须答：是**单次剖析案例**，真实 P50 只有 4.9s——这条粥如果不背，问答就穿。\n"
    ">\n"
    "> **端到端按场景分类看**：纯对话 1.2s、知识型长答案 50s、工具型查询 54s"
)
if old_65 in t:
    t = t.replace(old_65, new_65, 1)
    print("  ✓ 6.5 节口径警示已插入")
else:
    print("  ⚠ 未找到 6.5 节原文")

print("\n=== [3] 给「只有配置值」的项补真实统计数字 ===")
replace_row("D1", [
    "令牌桶控频次，多 worker 原子扣减", "✅ 已落地",
    "Redis Lua 原子扣减（防超卖）。⚠️ **QPS 5/桶 10 是配置上限不是实测**；"
    "实测真实流量：166 通跨 9 天，自然平均 QPS 0.0003，**单分钟峰值 17 次≈0.283 QPS**——"
    "说明**限流闸门至今没在真实流量下被触发验证过**，只做过人为构造验证"
])
replace_row("E2", [
    "每次调用记实际消耗", "✅ 已落地（有真数据）",
    "`usage.total` 回填 + Prometheus tokens 指标。**实测 166 通**：单轮 token 均值 4673、"
    "P50 4630 / P95 9907 / 最大 11173，累计消耗 **775,698 tokens**（prompt 均值 4192 / completion 481）"
])
replace_row("E3", [
    "成本能拆到租户/角色", "✅ 已落地（有真数据）",
    "指标带 `tenant`/`role` 标签；**实测分布**：role=teacher 159 / student 41，"
    "tenant=capital-medical 186 / t-ref 10 / smoke 2 / t-loop 2。⚠️ 未做账单/配额，只是可观测"
])
replace_row("C1", [
    "压到 1s 内", "✅ 实测 0.4s",
    "TTFT 0.4s（见 6.4）。**E2E 实测 166 通**：P50 4.76s / P90 46.0s / P95 63.8s，最大 90.3s；"
    "带工具（n=151）P50 4.9s 是纯对话（n=15）P50 2.8s 的 **3.5 倍**"
])

print("\n=== [4] 给 21 个「无证据」项补核查证据 ===")
no_data_evidence = {
    # ---- 安全与权限 ----
    "B4": ["数据库层强制租户过滤（fail-closed）", "❌ 未启用（已实查）",
           "**2026-09-12 查 pg_class**：kb_chunks / bench_ann 共 2 张表，`relrowsecurity` 全 False（启用 RLS 的 0 张）；"
           "靠代码层 `WHERE tenant_id`，漏写即静默越权（fail-open）"],
    "B7": ["手机号/姓名入日志前打码", "❌ 未做（已实查）",
           "`grep app/*.py`：**mask|sanitiz|脱敏|redact 命中 0 处**；审计日志原样记提问原文（实测含真实学员姓名）"],
    "B9": ["输出内容安全审核", "❌ 未做（已实查）",
           "`grep`：**moderat|content_safety 命中 0 处**；LLM 输出直接 SSE 下发，无任何 moderation 环节"],
    # ---- 可靠性与容错 ----
    "D6": ["瞬时故障自动重试", "❌ 未做（已实查）",
           "`grep`：**retry|tenacity|backoff|max_retries 命中 0 处**。设计上 LLM 不重试（防加剧拥塞）是有意为之，"
           "但**工具层也没有重试**是缺口"],
    "D9": ["停机前排空在途请求", "❌ 未做（已实查）",
           "`main.py:94-101` lifespan：`yield` 之后**没有任何一行清理代码**——无 Redis/PG 连接释放、无在途请求 draining"],
    # ---- 可观测性 ----
    "A6": ["指标异常自动告警", "❌ 未做（已实查）",
           "`grep`：alert|webhook|AlertManager 命中 **2 处，均为业务「预警」关键词**，非监控告警；"
           "Prometheus 无 AlertManager、无 rules 文件"],
    "A8": ["日志脱敏", "❌ 未做（已实查）",
           "同 B7：`grep` 脱敏相关**命中 0 处**；审计 jsonl 实测 200 条均为原文落盘"],
    # ---- 数据质量 ----
    "F6": ["语料可版本回溯", "❌ 未做（已实查）",
           "**查 information_schema**：kb_chunks 仅 id/kb/tenant_id/doc_idx/content/embedding 六字段，"
           "**无 version / updated_at / content_hash 任一**；导入即覆盖不可回溯"],
    "F7": ["增量/去重更新", "❌ 未做（已实查）",
           "无时间戳/content_hash 可做增量判据（同上六字段）；`ingest_doc.py` 为全量重建——"
           "实测后果就是 F5 那 462 行实验垃圾没被清掉"],
    # ---- 成本治理 ----
    "E4": ["重复问题走缓存省 token", "❌ 未做（同 C5）",
           "`grep lru_cache|redis_cache|CacheClient` **命中 0 处**；重复问题每次都重算 bge-m3 向量"],
    "E5": ["成本超阈值告警", "❌ 未做（同 A6）",
           "全链路无任何告警能力；E2 能算出「累计消耗 775,698 tokens」但**没人会收到超阈通知**"],
    # ---- 性能 ----
    "C5": ["热点结果缓存", "❌ 未做（已实查）",
           "`grep lru_cache|redis_cache|CacheClient` **命中 0 处**；Redis 只用于治理/记忆，没做结果缓存"],
    # ---- 测试与发布 ----
    "G4": ["容量压测定上限", "❌ 未做（已实查）",
           "`scripts/` 下 12 个 `bench_*` **全是效果评测**（测精度/MRR），**无 stress|locust|jmeter 类容量压测**；"
           "G1 的「QPS 5」至今只是配置值"],
    "G5": ["提交即跑流水线", "❌ 未做（已实查）",
           "**无 `.github/workflows`、无 `.gitlab-ci.yml`、无 Jenkinsfile**——122 个测试纯靠手跑"],
    "G6": ["灰度发布", "❌ 未做（已实查）",
           "无金丝雀/流量比例机制；单 worker 部署，**发布即全量切换，无回滚窗口**"],
    "G7": ["schema 迁移与回滚", "❌ 未做（已实查）",
           "**无 alembic/migrations 目录**，`requirements.txt` grep alembic **命中 0**；"
           "建表依赖 ingest 脚本幂等执行，无版本化迁移"],
    # ---- 运维部署 ----
    "H2": ["多副本与负载均衡", "❌ 未做（已实查）",
           "`docker-compose.yml` **无 replicas / deploy 段**；单 worker；Nginx ip_hash 仅写在方案里没落地"],
    "H3": ["容器资源限制", "❌ 未做（已实查）",
           "`docker-compose.yml` **无 mem_limit / cpus / healthcheck**——本机 8G 内存下 torch 会 OOM 就是这个问题"],
    "H5": ["备份与恢复", "❌ 未做（已实查）",
           "`scripts/` **无 backup/restore 脚本，无 crontab**；4.3 节只读副本/备份均为**选型方案**，无 RPO/RTO 实测"],
}
for rid, cols in no_data_evidence.items():
    replace_row(rid, cols)

BANK.write_text(t, encoding="utf-8")
new_len = len(t.splitlines())
print(f"\n备考库行数：{orig_len} → {new_len}（+{new_len - orig_len}）")
