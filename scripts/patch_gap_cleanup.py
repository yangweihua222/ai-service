# -*- coding: utf-8 -*-
"""集中后清理：标题项数修正 + 去掉 8 份重复引导 + 统一状态口径。"""
from pathlib import Path
import re

P = Path(r"D:\简历\技术点深挖备考库.md")
t = P.read_text(encoding="utf-8")

# ---------- 1. 域标题项数修正（按域内实际 ID 行数） ----------
lines = t.splitlines()
DOMAINS = {"B": "安全与权限", "D": "可靠性与容错", "A": "可观测性", "F": "数据质量",
           "E": "成本治理", "C": "性能", "G": "测试与发布", "H": "运维部署"}

# 逐域统计行数
counts = {}
cur = None
for l in lines:
    m = re.match(r"^## 域 ([A-H]) · ", l)
    if m:
        cur = m.group(1)
        counts[cur] = 0
        continue
    if cur and re.match(r"^\| [A-H]\d+ \|", l):
        counts[cur] += 1

n = 0
for ltr, nm in DOMAINS.items():
    cnt = counts.get(ltr, 0)
    # 旧标题里项数是任意值，用正则统一替换
    pat = re.compile(r"^## 域 " + ltr + r" · " + nm + r"（\d+ 项）", re.M)
    t, k = pat.subn(f"## 域 {ltr} · {nm}（{cnt} 项）", t)
    n += k
    print(f"  域 {ltr} {nm}: {cnt} 项 (替换 {k})")
print("标题修正数:", n)

# ---------- 2. 删除 8 个域里重复的「怎么用 / 状态口径」引导（章首已有一份） ----------
lines = t.splitlines()
REPEAT_HEAD = "**怎么用这张表**："
REPEAT_STATE = "**状态口径**："
out = []
i = 0
removed = 0
while i < len(lines):
    l = lines[i]
    if l.startswith(REPEAT_HEAD):
        # 吃掉连续引用块（到下一个空行为止）+ 紧随的状态口径行
        j = i
        while j < len(lines) and lines[j].strip() != "":
            j += 1
        # j 指向空行
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines) and lines[j].startswith(REPEAT_STATE):
            while j < len(lines) and lines[j].strip() != "":
                j += 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
        removed += 1
        i = j
        continue
    out.append(l)
    i += 1
print("删除重复引导块:", removed)
t = "\n".join(out)

# ---------- 3. 统一 🔶 口径 ----------
# 域内残留的独立状态口径行已删；此处同步章内其他表述
OLD = "| 🔶 规划中 | 技术上清楚，当前数据量用不上 | 讲清「什么时候才需要开」 |"
NEW = "| 🔶 量级未到 | 技术清楚、当前数据量用不上 | 讲清「什么时候才需要开」 |"
if OLD in t:
    t = t.replace(OLD, NEW)
    print("已统一 🔶 口径")

P.write_text(t, encoding="utf-8")
lines = t.splitlines()
print()
print("总行数:", len(lines))
print("域标题数:", len([l for l in lines if l.startswith("## 域")]))
print("60 项行数:", len([l for l in lines if re.match(r"^\| [A-H]\d+ \|", l)]))
print("残留重复引导:", t.count(REPEAT_HEAD))
