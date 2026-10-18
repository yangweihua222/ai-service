# -*- coding: utf-8 -*-
"""业务数据层：Text2SQL Schema + Mock 学员/轮转数据 + 轮转分配引擎（贪心+局部搜索）。

语义红线（与简历/备考库一致）：
  - 轮转 = 把学员分配到各科室（贪心 + 局部搜索）
  - 排班 = 考勤打卡班次计划（单独管理，本骨架聚焦轮转可行性检测）
对应简历「轮转分配模块」「AI 教学助手 · Text2SQL 学员数据」。

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点15 Text2SQL（自然语言查数据）       → 见 build_prompt() / generate_sql() / execute_sql() / summarize_rotation()
  · 技术点15 护栏：只读校验 + 租户强制过滤（不信任模型输出）→ 见 assert_sql_safe()
  · 技术点15 边界红线：学员档案是结构化数据，走 Text2SQL 而非 RAG → 见 summarize_rotation()
  · 技术点19 贪心+局部搜索 · 轮转分配算法      → 见 greedy_init() / objective() / local_search() / check_rotation_feasibility()
  · 技术点7/9 多租户数据隔离（行级 WHERE tenant_id） → 真实场景 SQL 注入处（本骨架 mock 不连库）
"""
import os
import random
import re
import sqlite3
from typing import List, Dict, Tuple, Optional

from app.config import TEACHING_DB_PATH

# ============ 1. Text2SQL Schema（注入到 Prompt）============
# 📌 技术点15：Schema 注入是 Text2SQL 的第一道地基——LLM 只能基于给定表结构生成，
#    同时明确要求"必须带 tenant_id 过滤"，把多租户约束前移到生成阶段。
SCHEMA = """
表 intern（实习生）
  id INT, tenant_id VARCHAR(租户/院校标识), name VARCHAR(姓名), dept VARCHAR(当前科室), period INT(当前期号)
表 rotation（轮转记录）
  id INT, tenant_id VARCHAR(租户/院校标识), intern_id INT, intern_name VARCHAR(姓名),
  dept VARCHAR(轮转科室), score_theory INT(理论成绩), score_skill INT(技能成绩), status VARCHAR(状态)
表 attendance（考勤打卡记录：谁哪天哪个班次、状态如何）
  id INT, tenant_id VARCHAR(租户/院校标识), intern_id INT, intern_name VARCHAR(姓名),
  work_date TEXT(打卡日期, 如 2026-04-01), shift VARCHAR(班次: 白班/夜班), status VARCHAR(状态: 正常/缺卡/请假)

约束：
1. 任何查询都必须带 tenant_id 过滤（多租户隔离，禁止跨院校查数据）。
2. 只允许 SELECT，禁止 INSERT/UPDATE/DELETE/DROP 等写操作。
"""

# ============ 2. 教学数据源 ============
# 已由"写死的 MOCK_ROTATION"升级为本地 SQLite 真实库（见 2.5 节 _SEED_ROWS / _ensure_db）。
# 生产切换：把 _connect() 换成 MySQL 驱动即可，执行层与业务层已隔离。


# ============ 2.5 真实教学库（本地 SQLite，首次调用自动建表 + 种子数据）============
# 📌 真实落地（2026-09-01）：原 mock_execute 返回写死的 MOCK_ROTATION，SQL 根本没执行；
#    现改为本地 SQLite 教学库真跑 SQL。SQLite 是 Python 内置、零依赖；生产切 MySQL
#    只需换 _connect() 的驱动（执行层与业务层已隔离）。
_SEED_ROWS: List[Tuple] = [
    # (tenant_id, intern_id, intern_name, dept, score_theory, score_skill, status)
    ("capital-medical", 1, "张三", "内科", 82, 78, "合格"),
    ("capital-medical", 1, "张三", "外科", 88, 85, "合格"),
    ("capital-medical", 1, "张三", "妇产科", 80, 83, "合格"),
    ("capital-medical", 1, "张三", "儿科", 85, 80, "合格"),
    ("capital-medical", 2, "李四", "内科", 91, 88, "合格"),
    ("capital-medical", 2, "李四", "外科", 76, 80, "合格"),
    ("capital-medical", 2, "李四", "妇产科", 84, 79, "合格"),
    ("capital-medical", 2, "李四", "儿科", 88, 92, "合格"),
    ("capital-medical", 3, "王五", "内科", 69, 72, "待改进"),
    ("capital-medical", 3, "王五", "外科", 74, 70, "待改进"),
    ("capital-medical", 3, "王五", "妇产科", 78, 75, "合格"),
    ("capital-medical", 3, "王五", "儿科", 80, 77, "合格"),
    ("capital-medical", 4, "赵六", "内科", 93, 90, "合格"),
    ("capital-medical", 4, "赵六", "外科", 89, 86, "合格"),
    ("capital-medical", 4, "赵六", "妇产科", 92, 94, "合格"),
    ("capital-medical", 4, "赵六", "儿科", 87, 85, "合格"),
    # 另一院校（租户）：用于验证多租户隔离——同样 SQL 换 tenant 查不到别人数据
    ("other-hospital", 11, "孙七", "内科", 77, 74, "合格"),
    ("other-hospital", 11, "孙七", "外科", 81, 79, "合格"),
    ("other-hospital", 12, "周八", "儿科", 66, 70, "待改进"),
    ("other-hospital", 12, "周八", "妇产科", 73, 71, "合格"),
]

# 考勤种子（2026-03 ~ 2026-04 打卡记录）：覆盖 正常/缺卡/请假 三种状态，供"考勤"查询演示
# 📌 语义红线：考勤=打卡班次（谁哪天该打卡），与"轮转=学员分配到科室"职责不同，绝不混用。
_ATTENDANCE_SEED: List[Tuple] = [
    # (tenant_id, intern_id, intern_name, work_date, shift, status)
    ("capital-medical", 1, "张三", "2026-03-02", "白班", "正常"),
    ("capital-medical", 1, "张三", "2026-03-03", "白班", "正常"),
    ("capital-medical", 1, "张三", "2026-03-04", "夜班", "正常"),
    ("capital-medical", 1, "张三", "2026-03-05", "夜班", "缺卡"),
    ("capital-medical", 1, "张三", "2026-03-06", "白班", "请假"),
    ("capital-medical", 2, "李四", "2026-03-02", "白班", "正常"),
    ("capital-medical", 2, "李四", "2026-03-03", "夜班", "正常"),
    ("capital-medical", 2, "李四", "2026-03-04", "夜班", "正常"),
    ("capital-medical", 2, "李四", "2026-03-05", "白班", "请假"),
    ("capital-medical", 2, "李四", "2026-03-06", "白班", "正常"),
    ("capital-medical", 3, "王五", "2026-03-02", "白班", "正常"),
    ("capital-medical", 3, "王五", "2026-03-03", "白班", "缺卡"),
    ("capital-medical", 3, "王五", "2026-03-04", "白班", "正常"),
    ("capital-medical", 3, "王五", "2026-03-05", "夜班", "正常"),
    ("capital-medical", 3, "王五", "2026-03-06", "夜班", "缺卡"),
    ("capital-medical", 4, "赵六", "2026-03-02", "夜班", "正常"),
    ("capital-medical", 4, "赵六", "2026-03-03", "夜班", "正常"),
    ("capital-medical", 4, "赵六", "2026-03-04", "白班", "正常"),
    ("capital-medical", 4, "赵六", "2026-03-05", "白班", "正常"),
    ("capital-medical", 4, "赵六", "2026-03-06", "夜班", "请假"),
    # 另一院校（租户）：隔离验证
    ("other-hospital", 11, "孙七", "2026-03-02", "白班", "正常"),
    ("other-hospital", 11, "孙七", "2026-03-03", "白班", "缺卡"),
    ("other-hospital", 12, "周八", "2026-03-02", "夜班", "正常"),
    ("other-hospital", 12, "周八", "2026-03-03", "夜班", "正常"),
]


def _connect():
    conn = sqlite3.connect(TEACHING_DB_PATH)
    conn.row_factory = sqlite3.Row  # 让结果能按列名取，直接转 dict
    return conn


def _ensure_db():
    """首次调用自动建库建表并灌入种子数据（幂等）。"""
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS intern (
              id INTEGER PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              name TEXT NOT NULL,
              dept TEXT,
              period INTEGER
            );
            CREATE TABLE IF NOT EXISTS rotation (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tenant_id TEXT NOT NULL,
              intern_id INTEGER NOT NULL,
              intern_name TEXT NOT NULL,
              dept TEXT NOT NULL,
              score_theory INTEGER,
              score_skill INTEGER,
              status TEXT
            );
            CREATE TABLE IF NOT EXISTS attendance (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              tenant_id TEXT NOT NULL,
              intern_id INTEGER NOT NULL,
              intern_name TEXT NOT NULL,
              work_date TEXT NOT NULL,
              shift TEXT NOT NULL,
              status TEXT NOT NULL
            );
            """
        )
        cur = conn.execute("SELECT COUNT(*) AS c FROM rotation")
        if cur.fetchone()["c"] == 0:
            conn.executemany(
                "INSERT INTO rotation (tenant_id, intern_id, intern_name, dept, score_theory, score_skill, status)"
                " VALUES (?,?,?,?,?,?,?)",
                _SEED_ROWS,
            )
            interns = {}
            for r in _SEED_ROWS:
                interns[(r[0], r[1])] = (r[2], r[3])
            conn.executemany(
                "INSERT INTO intern (id, tenant_id, name, dept, period) VALUES (?,?,?,?,?)",
                [(k[1], k[0], v[0], v[1], 1) for k, v in interns.items()],
            )
        cur = conn.execute("SELECT COUNT(*) AS c FROM attendance")
        if cur.fetchone()["c"] == 0:  # 考勤表独立判断（表可已存在但未灌数据）
            conn.executemany(
                "INSERT INTO attendance (tenant_id, intern_id, intern_name, work_date, shift, status)"
                " VALUES (?,?,?,?,?,?)",
                _ATTENDANCE_SEED,
            )
        conn.commit()
    finally:
        conn.close()


# ---------- SQL 安全闸门（面试加分点：Text2SQL 必须有护栏）----------
_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "attach", "pragma", "replace", "vacuum",
)


def _ast_check_tenant(sql: str) -> Optional[str]:
    """📌 P0 必修 (2026-09-08 code review): AST 解析强制 tenant_id 等值谓词。

    旧的字符串包含检查 `if "tenant_id" not in s` 可被字面量绕过
    （如 `WHERE 'tenant_id' = 'x'` 仍含子串），改用 sqlglot 做 AST 解析：
      ① WHERE 子句里必须有 `tenant_id = ?` 形式的等值谓词（? 在右侧）
      ② tenant_id 必须是**列引用**（Column 节点），不能是字符串字面量
      ③ 禁止子查询（防止 `(SELECT tenant_id FROM other_table)` 绕过）
      ④ 禁止多语句

    返回 None 表示通过；返回 str 表示拒绝原因。
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return "安全拦截：缺少 sqlglot 依赖（pip install sqlglot）"

    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        return f"安全拦截：SQL 解析失败（{e}）"

    # ① 禁止多语句
    if len(statements) != 1:
        return "安全拦截：禁止多语句执行"

    stmt = statements[0]
    if stmt is None:
        return "安全拦截：SQL 解析为空"

    # 必须是 SELECT
    if not isinstance(stmt, exp.Select):
        return "安全拦截：只允许 SELECT 只读查询"

    where = stmt.args.get("where")
    if where is None:
        return "安全拦截：缺少 WHERE 子句（无 tenant_id 过滤上下文）"

    # ③ 禁止子查询
    for node in where.find_all(exp.Subquery):
        return "安全拦截：禁止子查询（防止用子查询绕过租户过滤）"

    # ② + 检查 WHERE 里必须有 tenant_id = ? 形式
    found_tenant_eq = False
    for eq in where.find_all(exp.EQ):
        left, right = eq.left, eq.right
        # 形态 A：tenant_id = ? （左列、右占位）
        if (
            isinstance(left, exp.Column)
            and left.name.lower() == "tenant_id"
            and isinstance(right, exp.Placeholder)
        ):
            found_tenant_eq = True
            break
        # 形态 B：? = tenant_id （左占位、右列）—— 也算
        if (
            isinstance(right, exp.Column)
            and right.name.lower() == "tenant_id"
            and isinstance(left, exp.Placeholder)
        ):
            found_tenant_eq = True
            break

    if not found_tenant_eq:
        return "安全拦截：WHERE 子句必须包含 `tenant_id = ?` 等值谓词（防字面量绕过）"

    return None


def assert_sql_safe(sql: str, tenant_id: str) -> None:
    """执行前多道硬校验：① 只读（防误删/注入破坏）② 必须带 tenant_id（防跨租户越权）
    ③ 禁止子查询 ④ 禁止写死租户字面量（必须走 ? 占位）。

    📌 2026-09-08 code review P0 修复：用 sqlglot 做 AST 校验替代字符串包含判
    （原 `if "tenant_id" not in s` 可被字面量绕过）；保留字符串层硬检查作双保险。
    """
    s = sql.strip().lower().rstrip(";").strip()
    if ";" in s:
        raise ValueError("安全拦截：禁止多语句执行")
    if not s.startswith("select"):
        raise ValueError("安全拦截：只允许 SELECT 只读查询")
    for w in _FORBIDDEN:
        if re.search(rf"\b{w}\b", s):
            raise ValueError(f"安全拦截：禁止写操作 {w.upper()}")
    if not tenant_id:
        raise ValueError("安全拦截：缺少租户标识，拒绝执行")
    # 字符串层硬检查（双保险，快路径失败立即抛）
    if "tenant_id" not in s:
        raise ValueError("安全拦截：查询缺少 tenant_id 过滤，存在跨租户越权风险")
    if re.search(r"\(\s*select\b", s):
        raise ValueError("安全拦截：禁止子查询（防止用子查询绕过租户过滤）")
    if re.search(r"tenant_id\s*=\s*['\"]", s):
        raise ValueError("安全拦截：禁止写死租户标识（必须用 ? 占位符由系统绑定）")

    # AST 层硬检查（兜底，防字符串层漏检的字面量绕过）
    ast_err = _ast_check_tenant(sql)
    if ast_err:
        raise ValueError(ast_err)


def execute_sql(sql: str, tenant_id: str = "") -> List[Dict]:
    """真实执行 SQL（SQLite），返回行字典列表。tenant 值用参数绑定，杜绝拼接注入。"""
    _ensure_db()
    assert_sql_safe(sql, tenant_id)
    params = [tenant_id] * sql.count("?")
    conn = _connect()
    try:
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ============ 3. Text2SQL（规则生成 SQL；真实场景由 LLM 完成）============
def build_prompt(question: str) -> str:
    return f"""你是一个 SQL 生成器。数据库 schema 如下：
{SCHEMA}
要求：
1. 只输出一条可执行的 SQL，不要任何解释、不要 markdown 代码块符号。
2. 表名/字段名必须来自上面的 schema。
3. 租户标识用占位符 ? 表示（执行时由系统绑定），不要写死 tenant_id 的值。
4. 用户问题：{question}
"""


def generate_sql(question: str, tenant_id: str = "") -> str:
    """Mock LLM（关键词规则生成 SQL）。真实模式：调 LLM.chat(build_prompt(q))。
    采用**结构化拼装**（select/where/group/order/limit 分离）而非字符串追加——
    早期版本在 GROUP BY 之后追加 WHERE，生成的 SQL 根本无法执行，真连库后才暴露。
    📌 技术点15 Text2SQL：SCHEMA 注入 + 输出约束(只输出一条可执行 SQL) +
       租户约束前移(生成阶段就带 tenant_id = ?) + 评测集驱动迭代（见 eval_text2sql）。"""
    q = question
    where = ["tenant_id = ?"]          # 多租户约束前移到生成阶段
    select = "*"
    group = order = limit = ""

    # 表路由（语义红线）：考勤类问题 → attendance（打卡班次），其余 → rotation（轮转成绩）。
    #   二者职责不同：考勤=谁哪天哪个班次/状态，轮转=学员分配到哪些科室。绝不混查。
    is_attendance = any(k in q for k in ("考勤", "打卡", "出勤", "缺卡", "补卡"))
    table = "attendance" if is_attendance else "rotation"

    # 学员姓名过滤（真 LLM 场景关键）：问"张三去哪了"必须生成精确 SQL，
    #   否则全表返回 + 统计摘要，LLM 拿不到明细、答不出具体去向。
    name_m = re.search(r"(张三|李四|王五|赵六|孙七|周八)", q)
    if name_m:
        where.append(f"intern_name='{name_m.group(1)}'")

    if is_attendance:
        # 考勤状态过滤（正常/缺卡/请假）
        if "缺卡" in q:
            where.append("status='缺卡'")
        elif "请假" in q:
            where.append("status='请假'")
        elif "正常" in q or "出勤" in q:
            where.append("status='正常'")
    else:
        need_avg = ("平均" in q) or ("成绩" in q) or ("分" in q)
        if need_avg:
            if "理论" in q:
                select = "intern_id, intern_name, AVG(score_theory) AS avg_theory"
            elif "技能" in q:
                select = "intern_id, intern_name, AVG(score_skill) AS avg_skill"
            else:
                select = "intern_id, intern_name, AVG((score_theory+score_skill)/2) AS avg_total"
            group = " GROUP BY intern_id, intern_name"

        m = re.search(r"(妇产科|内科|外科|儿科)", q)   # 注意"妇产科"要排在"外科"前，否则被截断匹配
        if m:
            where.append(f"dept='{m.group(1)}'")

        if "不合格" in q or "待改进" in q:              # "不合格"含"合格"子串，必须先判
            where.append("status='待改进'")
        elif "合格" in q or "状态" in q:
            where.append("status='合格'")

        if "最高" in q or "最低" in q:
            col = "avg_theory" if "理论" in q else ("avg_skill" if "技能" in q else "avg_total")
            if select == "*":  # 非聚合场景按单科成绩排序
                col = "score_theory" if "理论" in q else ("score_skill" if "技能" in q else "(score_theory+score_skill)/2")
            order = f" ORDER BY {col} DESC" if "最高" in q else f" ORDER BY {col} ASC"

        if re.search(r"前\s*(三|3)|top\s*3", q, re.I):
            limit = " LIMIT 3"

    return f"SELECT {select} FROM {table} WHERE " + " AND ".join(where) + group + order + limit + ";"


def summarize_rotation(rows: List[Dict]) -> str:
    """结构化数据 → 学员成长档案智能摘要（真实场景由 LLM 总结；此处按结果形态轻量聚合）。
    📌 技术点15 边界红线：学员成长档案是结构化数据(轮转/分数/评语在表里)，
       正确路线是 Text2SQL 查出数据 → LLM 写摘要，绝不走 RAG。
    ⚙️ 真实执行后结果有两种形态：聚合查询（含 avg_* 列）与明细查询，需分别适配。"""
    if not rows:
        return "未查询到相关记录。"

    keys = set(rows[0].keys())
    avg_cols = [k for k in keys if k.startswith("avg_")]
    label_map = {"avg_theory": "理论", "avg_skill": "技能", "avg_total": "综合"}

    if avg_cols:  # 聚合型：按学员展示平均分
        col = avg_cols[0]
        label = label_map.get(col, col)
        parts = [
            f"{r.get('intern_name') or ('学员' + str(r.get('intern_id')))} {float(r[col]):.1f}"
            for r in rows
        ]
        return f"共 {len(rows)} 名学员，{label}平均成绩：" + "；".join(parts) + " 分。"

    if {"dept", "score_theory", "score_skill"} <= keys:  # 明细型
        n = len(rows)
        ok_n = sum(1 for r in rows if r.get("status") == "合格")
        avg_t = sum(r["score_theory"] for r in rows) / n
        avg_s = sum(r["score_skill"] for r in rows) / n
        names = sorted({str(r.get("intern_name") or r.get("name") or "") for r in rows} - {""})
        who = "、".join(names[:5]) + (" 等" if len(names) > 5 else "")
        head = (
            f"共 {n} 条轮转记录（{who}）：理论平均 {avg_t:.1f} 分，"
            f"技能平均 {avg_s:.1f} 分，合格 {ok_n}/{n}。"
        )
        # 📌 真 LLM 场景：只给统计摘要，LLM 答不出"张三去哪了"这类明细问题；
        #    行数少(≤8)直接给明细行，行数多给前 5 条并提示精查条件，控制上下文长度。
        def _fmt(r) -> str:
            name = r.get("intern_name") or r.get("name") or ("学员" + str(r.get("intern_id")))
            return f"  {name} | {r.get('dept')} | 理论{r.get('score_theory')} 技能{r.get('score_skill')} | {r.get('status')}"

        if n <= 8:
            return head + "\n明细：\n" + "\n".join(_fmt(r) for r in rows)
        return head + "\n明细（前 5 条，如需精确结果请指定姓名/科室/成绩条件）：\n" + "\n".join(
            _fmt(r) for r in rows[:5]
        )

    return f"查询到 {len(rows)} 条记录。"


def summarize_attendance(rows: List[Dict]) -> str:
    """考勤（打卡班次）数据 → 可读摘要。
    📌 语义红线（讲错必翻车）：考勤=谁哪天哪个班次(白班/夜班)、状态(正常/缺卡/请假)，
       与"轮转=学员分配到科室"职责不同，绝不混用、绝不写成"轮转排班"。
    ⚙️ 与 summarize_rotation 对称：行数少(≤8)给完整明细（真 LLM 场景下缺了明细，
       LLM 答不出"张三缺卡记录"这类问题）；行数多给状态统计 + 前 5 条提示精查。"""
    if not rows:
        return "未查询到相关记录。"

    keys = set(rows[0].keys())
    if {"work_date", "shift", "status"} <= keys:  # 明细型（attendance 表返回形态）
        n = len(rows)
        status_cnt = {}
        for r in rows:
            status_cnt[r.get("status") or "未知"] = status_cnt.get(r.get("status") or "未知", 0) + 1
        names = sorted({str(r.get("intern_name") or "") for r in rows} - {""})
        who = "、".join(names[:5]) + (" 等" if len(names) > 5 else "")
        stat = "、".join(f"{k}{v}条" for k, v in sorted(status_cnt.items()))
        head = f"共 {n} 条考勤记录（{who}）：{stat}。"

        def _fmt(r) -> str:
            name = r.get("intern_name") or ("学员" + str(r.get("intern_id")))
            return f"  {name} | {r.get('work_date')} | {r.get('shift')} | {r.get('status')}"

        if n <= 8:
            return head + "\n明细：\n" + "\n".join(_fmt(r) for r in rows)
        return head + "\n明细（前 5 条，如需精确结果请指定姓名/日期/状态条件）：\n" + "\n".join(
            _fmt(r) for r in rows[:5]
        )

    return f"查询到 {len(rows)} 条考勤记录。"


# ============ 4. 轮转分配引擎（学员 → 科室）：贪心 + 局部搜索 ============
interns = ["实习生A", "实习生B", "实习生C", "实习生D", "实习生E", "实习生F"]
rotation_plans = {
    "实习生A": ["内科", "外科", "妇产科", "儿科"],
    "实习生B": ["内科", "儿科", "外科", "妇产科"],
    "实习生C": ["外科", "内科", "儿科", "妇产科"],
    "实习生D": ["妇产科", "内科", "外科", "儿科"],
    "实习生E": ["儿科", "外科", "内科", "妇产科"],
    "实习生F": ["内科", "妇产科", "外科", "儿科"],
}
capacity = {"内科": 2, "外科": 2, "妇产科": 2, "儿科": 2}
total_periods = 5


def greedy_init():
    # 📌 技术点19 贪心初始化：每期按"剩余轮转需求越紧迫越优先"安排，容量满则顺延 → 快速拿到可行解
    schedule = [{name: None for name in interns} for _ in range(total_periods)]
    progress = {name: 0 for name in interns}
    for p in range(total_periods):
        used = {d: 0 for d in capacity}
        order = sorted(interns, key=lambda n: len(rotation_plans[n]) - progress[n], reverse=True)
        for name in order:
            ptr = progress[name]
            if ptr >= len(rotation_plans[name]):
                continue
            dept = rotation_plans[name][ptr]
            if used[dept] < capacity[dept]:
                schedule[p][name] = dept
                used[dept] += 1
                progress[name] += 1
    return schedule, progress


def objective(schedule):
    # 📌 技术点19 目标函数 = 未满足需求×100 + 负载方差（未满足硬指标权重最高，确保优先排满；方差软指标保均衡）
    unmet = 0
    for name in interns:
        done = sum(1 for p in schedule if p[name] is not None)
        unmet += max(0, len(rotation_plans[name]) - done)
    loads = {d: 0 for d in capacity}
    for p in schedule:
        for name, d in p.items():
            if d is not None:
                loads[d] += 1
    mean = sum(loads.values()) / len(loads)
    variance = sum((v - mean) ** 2 for v in loads.values()) / len(loads)
    return unmet * 100 + variance


def local_search(schedule, iterations=3000):
    # 📌 技术点19 局部搜索：随机交换两学员同期的科室，目标函数下降才接受(爬山)，迭代收敛；可升级模拟退火/禁忌搜索破局部最优
    best = objective(schedule)
    for _ in range(iterations):
        p = random.randrange(total_periods)
        i = random.choice(interns)
        j = random.choice(interns)
        if i == j:
            continue
        schedule[p][i], schedule[p][j] = schedule[p][j], schedule[p][i]
        score = objective(schedule)
        if score < best:
            best = score
        else:
            schedule[p][i], schedule[p][j] = schedule[p][j], schedule[p][i]
    return best


def check_rotation_feasibility() -> str:
    """轮转/排班可行性检测工具的核心逻辑：检测轮转分配缺口与负载均衡。

    排班（考勤打卡班次）单独管理，此处聚焦「学员→科室」轮转分配可行性。
    📌 语义红线（讲错必翻车）：轮转=把学员分配到各科室(贪心+局部搜索)；排班=考勤打卡班次计划(单独管理)。
       对应简历项目一「轮转分配模块」与技术点19。
    """
    random.seed(7)
    schedule, _ = greedy_init()
    before = objective(schedule)
    after = local_search(schedule)
    unmet = 0
    for name in interns:
        done = sum(1 for p in schedule if p[name] is not None)
        unmet += max(0, len(rotation_plans[name]) - done)
    advice = "轮转分配可行，无缺口" if unmet == 0 else f"存在 {unmet} 步轮转缺口，建议增大期数或放宽科室容量"
    return (
        f"轮转分配可行性检测：贪心初始化目标分 {before} → 局部搜索后 {after}（越低越好）；"
        f"{advice}。各科室容量={capacity}，总期数={total_periods}。"
    )


# ============ 5. Text2SQL 评测集（驱动迭代；简历口径：5/5 命中）============
# gold 以"关键要素"表达，评测时逐项判定（比整串文本比对更稳健，也容忍列别名差异）
TEXT2SQL_TESTSET = [
    ("查询所有实习生的理论平均成绩", "SELECT intern_id, intern_name, AVG(score_theory) AS avg_theory FROM rotation WHERE tenant_id = ? GROUP BY intern_id, intern_name;"),
    ("查询内科轮转的实习生", "SELECT * FROM rotation WHERE tenant_id = ? AND dept='内科';"),
    ("查询轮转合格的记录", "SELECT * FROM rotation WHERE tenant_id = ? AND status='合格';"),
    ("查询理论成绩最高的前三名", "SELECT intern_id, intern_name, AVG(score_theory) AS avg_theory FROM rotation WHERE tenant_id = ? GROUP BY intern_id, intern_name ORDER BY avg_theory DESC LIMIT 3;"),
    # 考勤域（2026-09-01 新增，简历 [49]③ 口径：Text2SQL 覆盖学员结构化档案含考勤）
    ("查询张三的缺卡记录", "SELECT * FROM attendance WHERE tenant_id = ? AND intern_name='张三' AND status='缺卡';"),
]


def _judge(pred: str, gold: str) -> bool:
    """按关键要素判定生成 SQL 是否正确（结构要素 + 多租户约束必须存在）。"""
    p = pred.lower().replace(" ", "")
    g = gold.lower().replace(" ", "")
    checks = ["tenant_id=?" in p]  # 无论什么问题，租户过滤都是硬要求
    # (gold 中出现即要求 pred 也出现的要素)
    tokens = [
        "avg(score_theory)", "avg(score_skill)", "groupby",
        "dept='内科'", "status='合格'", "limit3", "orderbyavg_theorydesc",
        "attendance", "intern_name='张三'", "status='缺卡'",
    ]
    for t in tokens:
        if t in g:
            checks.append(t in p)
    return all(checks)


def eval_text2sql(tenant_id: str = "capital-medical") -> dict:
    """评测：既判 SQL 结构正确性，也**真实执行验证能否跑通并返回行**。
    📌 真实落地的意义：以前只比对 SQL 字符串，语法错了也发现不了；
       现在每条都在本地库真跑一遍，"生成对"和"能跑通"双重校验。"""
    hit = 0
    ran = 0
    details = []
    for q, gold in TEXT2SQL_TESTSET:
        pred = generate_sql(q, tenant_id)
        ok = _judge(pred, gold)
        rows, err = [], None
        try:
            rows = execute_sql(pred, tenant_id)
        except Exception as e:  # 生成了危险 SQL 或语法错误 → 直接暴露在评测里
            err = str(e)
        ok_run = err is None and len(rows) > 0
        hit += 1 if ok else 0
        ran += 1 if ok_run else 0
        details.append({
            "query": q,
            "pred": pred,
            "hit": ok,
            "executed": ok_run,
            "rows": len(rows),
            "error": err,
        })
    n = len(TEXT2SQL_TESTSET)
    return {
        "total": n,
        "hit": hit,
        "rate": f"{hit / n * 100:.0f}%",
        "executed": ran,
        "exec_rate": f"{ran / n * 100:.0f}%",
        "details": details,
    }


def check_tenant_isolation() -> dict:
    """多租户隔离验证：同一条 SQL 换 tenant，只能看到本院校数据。
    📌 技术点7/9：Text2SQL 场景下租户隔离靠"生成阶段注入 tenant_id 条件 + 执行阶段参数绑定"，
       而不是靠模型自觉——这里给出可跑的证据。"""
    sql = generate_sql("查询轮转合格的记录", "capital-medical")
    a = execute_sql(sql, "capital-medical")
    b = execute_sql(sql, "other-hospital")
    tenants_a = {r["tenant_id"] for r in a}
    tenants_b = {r["tenant_id"] for r in b}
    return {
        "sql": sql,
        "capital-medical_rows": len(a),
        "other-hospital_rows": len(b),
        "tenants_a": sorted(tenants_a),
        "tenants_b": sorted(tenants_b),
        "isolated": tenants_a == {"capital-medical"} and tenants_b == {"other-hospital"},
    }
