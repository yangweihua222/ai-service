# -*- coding: utf-8 -*-
"""把散插在 8 个章节的「生产差距对照表」集中成独立一整章。

用户要求（2026-09-12）：同类内容集中成章，不要散插，便于检查有没有偷工减料。
做法：从原位置摘除 → 按域顺序重组 → 插到「从 0 到上线全流程（Stage 14）」之后。
本脚本只搬家 + 加引导，不改变任何表格行的内容。
"""
from pathlib import Path
import re

BANK = Path(r"D:\简历\技术点深挖备考库.md")
text = BANK.read_text(encoding="utf-8")
lines = text.splitlines()

# ---------- 1. 定位 8 个块的起止（含前置空行的回溯） ----------
idx = [i for i, l in enumerate(lines) if "生产差距对照表" in l]
blocks = []
for s in idx:
    st = s
    while st > 0 and lines[st - 1].strip() == "":
        st -= 1
    e = s + 1
    while e < len(lines):
        ls = lines[e]
        if re.match(r"^#{1,3} ", ls) and e > s + 1:
            break
        e += 1
    while e > 0 and lines[e - 1].strip() == "":
        e -= 1
    blocks.append((st, e))

names = []
contents = []
for st, e in blocks:
    head = lines[st]
    m = re.search(r"生产差距对照表：(.+?)（(\d+) 项）", head)
    names.append((m.group(1), int(m.group(2))) if m else ("?", 0))
    contents.append(lines[st:e])
    print(f"摘出块 [{st+1},{e}] {head[:40]}")

# 域顺序 + 标题映射（按 seq：安全→可靠→可观测→数据质量→成本→性能→测试发布→运维）
PLAN = [
    ("可观测性",    8, "**链路看得见吗**：出问题能不能知道、能不能定位。"),
    ("安全与权限", 10, "**进得来的人对不对**：身份、权限、租户、内容安全。"),
    ("可靠性与容错", 10, "**扛不扛得住**：限流、熔断、重试、停机、幂等。"),
    ("性能",        7,  "**快不快**：TTFT、流式、上下文、缓存、连接池、ANN。"),
    ("成本治理",    5,  "**花多少钱**：预算、计量、归因、缓存省成本、超阈告警。"),
    ("数据质量",    7,  "**喂进去的东西干不干净**：语料、分块、评测集、脏数据、版本、增量。"),
    ("测试与发布",  8,  "**改坏了能不能发现**：CI、压测、灰度、回滚、评测门禁。"),
    ("运维部署",    5,  "**能不能交给运维**：容器、副本、资源限制、密钥注入、备份。"),
]

by_name = {}
for (nm, cnt), body in zip(names, contents):
    by_name[nm] = (cnt, body)

new_lines = []
new_lines.append("")
new_lines.append("---")
new_lines.append("")
new_lines.append("# 生产差距 60 项对照·设计基线 vs 我们的现状")
new_lines.append("")
new_lines.append("> **这张表和上一张全流程表是一对**：上一章讲「应有的设计全貌」，这一章回答「**我们做到哪一步、差在哪**」。")
new_lines.append("> 8 个域 60 项，全部基于 2026-09-12 逐项 grep 代码 / 查库 / 看日志核实，**没做的就写没做**。")
new_lines.append(">")
new_lines.append("> **状态口径（四档，面试按这个报）**：")
new_lines.append(">")
new_lines.append("| 状态 | 含义 | 被追问时怎么答 |")
new_lines.append("| --- | --- | --- |")
new_lines.append("| ✅ 已落地 | 有代码、有测试、有证据 | 直接给代码位置 + 实测数字 |")
new_lines.append("| ⚠️ 有但带缺陷 / 刚修复 / 量级未到 | 做了一半或有前提条件 | **先说缺陷再给下一步**，别粉饰 |")
new_lines.append("| 🔶 量级未到 | 技术上清楚，但当前数据量用不上 | 讲清「什么时候才需要开」 |")
new_lines.append("| ❌ 未做 | 代码里确实没有 | **直说 + 给补齐方案**，❗禁止把规划说成已实现 |")
new_lines.append(">")
new_lines.append("> **数据口径二分制（这是全文统一规定）**：")
new_lines.append(">")
new_lines.append("> - **【实测】** = 真跑出来的数字，**可以报数、可以背**")
new_lines.append("> - **【取值】** = 合理估算，**旁边写明取值依据**；被问「凭什么」答依据，**不答「我实测过」**")
new_lines.append("")
new_lines.append("**速览（8 域 60 项一句话版，背下来能撑住 30 秒）**：")
new_lines.append("")
new_lines.append("| 域 | 项数 | 一句话现状 |")
new_lines.append("| --- | --- | --- |")
new_lines.append("| 可观测性 | 8 | 埋点、追踪、指标、审计**都有**，**缺告警规则**和日志脱敏 |")
new_lines.append("| 安全与权限 | 10 | 认证/授权/租户/注入/参数校验**都有**，**缺 RLS、脱敏、输出内容安全** |")
new_lines.append("| 可靠性与容错 | 10 | 限流/熔断/降级/超时/防循环**都有**，**缺重试、幂等、优雅停机** |")
new_lines.append("| 性能 | 7 | 流式/裁剪/路由**有**，**缺缓存**；ANN 属量级未到 |")
new_lines.append("| 成本治理 | 5 | 预算闸门/计量/归因**有**，**缺缓存省成本、成本告警** |")
new_lines.append("| 数据质量 | 7 | 评测集/评测体系**有**，**语料极少、脏数据没清、无版本无增量** |")
new_lines.append("| 测试与发布 | 8 | 单测/回归/评测门禁**有**，**缺 CI、压测、灰度、迁移回滚** |")
new_lines.append("| 运维部署 | 5 | 容器化**有**，**缺多副本、资源限制、Secret 管理、备份** |")
new_lines.append("")

total = 0
for nm, cnt, desc in PLAN:
    if nm not in by_name:
        print(f"  !! 缺少域：{nm}")
        continue
    real_cnt, body = by_name[nm]
    total += real_cnt
    new_lines.append("---")
    new_lines.append("")
    letter = re.match(r"^\| ([A-H])\d+ \|", body[2]).group(1) if len(body) > 2 else "?"
    new_lines.append(f"## 域 {letter} · {nm}（{real_cnt} 项）")
    new_lines.append("")
    new_lines.append(desc)
    new_lines.append("")
    new_lines.extend(body)
    new_lines.append("")

print(f"\n重组完成，合计 {total} 项")

# ---------- 2. 从后往前删除原位置的块 ----------
for st, e in sorted(blocks, reverse=True):
    del lines[st:e]

# ---------- 3. 插到 Stage 14 之后 ----------
anchor = None
for i, l in enumerate(lines):
    if l.startswith("## Stage 14"):
        # 找到该 Stage 之后的下一个一级标题或分隔线
        for j in range(i + 1, len(lines)):
            if lines[j].startswith("# "):
                anchor = j
                break
        break
if anchor is None:
    raise SystemExit("找不到 Stage 14 之后的插入锚点")

# 往前吃掉紧邻的空行和分隔线，保持整洁
while anchor > 0 and (lines[anchor - 1].strip() == "" or lines[anchor - 1].strip() == "---"):
    anchor -= 1

lines[anchor:anchor] = new_lines
BANK.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"已插入，锚点行 {anchor+1}")
print("新总行数:", len(lines))
