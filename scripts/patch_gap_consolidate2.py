# -*- coding: utf-8 -*-
"""把散插在 8 个章节的「生产差距对照表」集中成独立一整章（v2 纯位置映射版）。

用户要求（2026-09-12）：同类内容集中成章，不要分散，便于检查有没有偷工减料。
v2 修正：不再用中文正则解析域名（改为按文件出现顺序映射），不再回溯块起点。
本脚本只搬家 + 加引导，不改变任何表格行的内容。
"""
from pathlib import Path
import re

BANK = Path(r"D:\简历\技术点深挖备考库.md")
lines = BANK.read_text(encoding="utf-8").splitlines()

KEY = "生产差距对照表"
idx = [i for i, l in enumerate(lines) if KEY in l]
assert len(idx) == 8, f"期望 8 个块，实际 {len(idx)}"

# ---------- 1. 划定块范围：从标题行到下一个 # 级标题前 ----------
blocks = []
for i, s in enumerate(idx):
    end = len(lines)
    for j in range(s + 1, len(lines)):
        if re.match(r"^#{1,3} ", lines[j]):
            end = j
            break
    if i + 1 < len(idx) and idx[i + 1] < end:
        end = idx[i + 1]
    blocks.append((s, end))

# ---------- 2. 按文件出现顺序映射域名（不依赖中文解析） ----------
ORDER = [
    ("B", "安全与权限",   "进得来的人对不对：身份、权限、租户、注入、脱敏、内容安全。"),
    ("D", "可靠性与容错", "扛不扛得住：限流、预算、熔断、降级、超时、重试、幂等、停机。"),
    ("A", "可观测性",     "出问题知不知道、能不能定位：日志、追踪、指标、审计、告警。"),
    ("F", "数据质量",     "喂进去的东西干不干净：语料、分块、评测集、脏数据、版本、增量。"),
    ("E", "成本治理",     "花多少钱：预算闸门、计量、按租户归因、缓存省成本、超阈告警。"),
    ("C", "性能",         "快不快：TTFT、流式、上下文裁剪、缓存、连接池、检索 ANN。"),
    ("G", "测试与发布",   "改坏了能不能拦住：单测、e2e、回归、压测、CI、灰度、回滚、评测门禁。"),
    ("H", "运维部署",     "能不能交给运维：容器、多副本、资源限制、密钥注入、备份恢复。"),
]

bodies = []
for i, (s, end) in enumerate(blocks):
    head = lines[s]
    print(f"[{i}] 行{s+1}-{end}  {head[:52]}")
    bodies.append(lines[s:end])

# 校验：每个块的首个表格行首列字母应与计划一致
for (ltr, nm, _), body in zip(ORDER, bodies):
    first = next((l for l in body if re.match(r"^\| [A-H]\d+ \|", l)), None)
    got = first[2] if first else "?"
    flag = "OK" if got == ltr else f"!! 期望 {ltr}"
    print(f"  域 {ltr} {nm}: 表格首字母 {got} {flag}")
    assert got == ltr, f"表格首字母不匹配：期望 {ltr} 实得 {got}"

# ---------- 3. 组装新章节 ----------
new = []
new.append("")
new.append("---")
new.append("")
new.append("# 生产差距 60 项对照·设计基线 vs 我们的现状")
new.append("")
new.append("> **这一章和上一章是一对**：上一章「从 0 到上线全流程」讲**应有的设计全貌**，")
new.append("> 这一章回答**我们实际做到哪一步、差在哪**。8 个域 60 项，")
new.append("> 全部基于 2026-09-12 逐项 grep 代码 / 连库查表 / 翻日志核实，**没做的就写没做**。")
new.append("")
new.append("**状态口径（四档，面试按这个报，别混）**：")
new.append("")
new.append("| 状态 | 含义 | 被追问时怎么答 |")
new.append("| --- | --- | --- |")
new.append("| ✅ 已落地 | 有代码 + 有测试 + 有证据 | 给代码位置 + 实测数字 |")
new.append("| ⚠️ 有缺陷 / 刚修复 | 做了一半，或有前提条件 | **先说缺陷再说下一步**，别粉饰 |")
new.append("| 🔶 量级未到 | 技术清楚，当前数据量用不上 | 讲清「什么时候才需要开」 |")
new.append("| ❌ 未做 | 代码里确实没有 | **直说 + 给补齐方案**；❗禁止把规划说成已实现 |")
new.append("")
new.append("**数据口径二分制（全文统一规定）**：")
new.append("")
new.append("- **【实测】** = 真跑出来的数字，**可以报数、可以背**")
new.append("- **【取值】** = 合理估算，**旁边写明取值依据**；被问「凭什么」答依据，**不答「我实测过」**")
new.append("")
new.append("**8 域速览（背下来能撑 30 秒）**：")
new.append("")
new.append("| 域 | 项数 | 一句话现状 |")
new.append("| --- | --- | --- |")
new.append("| A 可观测性 | 8 | 日志/追踪/指标/审计**都有**，**缺告警规则和日志脱敏** |")
new.append("| B 安全与权限 | 10 | 认证/授权/租户/注入/参数校验**都有**，**缺 RLS、脱敏、输出内容安全** |")
new.append("| C 性能 | 7 | 流式/裁剪/路由**有**，**缺缓存**；ANN 属量级未到 |")
new.append("| D 可靠性与容错 | 10 | 限流/熔断/降级/超时/防循环**都有**，**缺重试、幂等、优雅停机** |")
new.append("| E 成本治理 | 5 | 预算/计量/归因**有**，**缺缓存省成本、成本告警** |")
new.append("| F 数据质量 | 7 | 评测体系**有**，**语料极少、脏数据没清、无版本无增量** |")
new.append("| G 测试与发布 | 8 | 单测/回归/评测门禁**有**，**缺 CI、压测、灰度、迁移回滚** |")
new.append("| H 运维部署 | 5 | 容器化**有**，**缺多副本、资源限制、Secret 管理、备份** |")
new.append("")

for (ltr, nm, desc), body in zip(ORDER, bodies):
    new.append("---")
    new.append("")
    new.append(f"## 域 {ltr} · {nm}（{len([l for l in body if re.match(r'^\\| [A-H]\\d+ \\|', l)])} 项）")
    new.append("")
    new.append(desc)
    new.append("")
    # 去掉原块自带的 ### 标题行，改成本章二级标题
    for j, l in enumerate(body):
        if j == 0 and KEY in l:
            continue
        new.append(l)
    new.append("")

# ---------- 4. 重建全文：跳过原块位置 ----------
skip = set()
for s, end in blocks:
    # 顺带吃掉块之前的连续空行
    st = s
    while st > 0 and lines[st - 1].strip() == "":
        st -= 1
    for k in range(st, end):
        skip.add(k)

kept = [l for i, l in enumerate(lines) if i not in skip]

# ---------- 5. 插到 Stage 14 之后 ----------
anchor = None
for i, l in enumerate(kept):
    if l.startswith("## Stage 14"):
        for j in range(i + 1, len(kept)):
            if kept[j].startswith("# "):
                anchor = j
                break
        break
assert anchor is not None, "找不到 Stage 14 之后的插入锚点"
while anchor > 0 and (kept[anchor - 1].strip() in ("", "---")):
    anchor -= 1

kept[anchor:anchor] = new
BANK.write_text("\n".join(kept) + "\n", encoding="utf-8")
print()
print(f"已插入，锚点行 {anchor+1}｜新总行数 {len(kept)}")
print("对照表数（应 8）:", sum(1 for l in kept if KEY in l))
print("60 项行数（应 60）:", len([l for l in kept if re.match(r"^\| [A-H]\d+ \|", l)]))
