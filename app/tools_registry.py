# -*- coding: utf-8 -*-
"""工具注册表：每个工具一份 JSON Schema + 权限 + 实现。

这正是简历里「工具集（Function Calling / Tool Use）」的落地：
每个工具一份 schema，供 Agent 调度与 MCP Server 复用同一份定义。

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点4 工具定义（JSON Schema：name/description/parameters） → 见 ToolSpec / openai_tools()
  · 技术点5 / 实战⑫ MCP 复用同一份工具契约（Agent 与 MCP Server 共享） → 见 registry 单例被 mcp_server 复用
  · 技术点8 RBAC 工具级可见性（按角色返回工具子集）             → 见 for_role()
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from typing import Optional

import jsonschema

from app.models import Role


def validate_args(spec: "ToolSpec", arguments: dict) -> Optional[str]:
    """📌 参数校验（第二道防线）：用工具自带的 JSON Schema 校验模型传的参数。

    返回 None 表示通过；否则返回人话错误描述（回灌 observation 让模型自我修正）。
    📌 设计取舍：本地校验而非信任模型——Schema 只约束模型输出格式，
    执行前必须再校验一次，防止幻觉/格式漂移导致工具拿到脏参数。
    """
    try:
        jsonschema.validate(instance=arguments or {}, schema=spec.parameters)
        return None
    except jsonschema.ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path)
        return f"{e.message}" + (f"（字段：{path}）" if path else "")
    except jsonschema.SchemaError:  # 工具自身 schema 写错 → 不阻塞调用，降级放行
        return None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    required: List[str] = field(default_factory=list)
    # 允许调用该工具的角色；默认全部角色（最终权限仍由 RBAC 二次校验）
    allowed_roles: List[Role] = field(default_factory=lambda: list(Role))
    execute: Optional[Callable] = None


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        # 📌 生产加固（2026-09-11 实测发现后补）：默认**拒绝未声明字段**。
        #    JSON Schema 不写 additionalProperties 时，模型多吐出来的幻觉参数会被静默放行，
        #    一路带到业务层才炸（或更糟：按默认值静默执行了错误的写操作）。
        #    实测：加固前「幻觉字段」拦截率 0/9，加固后 9/9（见 scripts/bench_engineering.py）。
        params = spec.parameters
        if isinstance(params, dict) and params.get("type") == "object" and "additionalProperties" not in params:
            spec.parameters = {**params, "additionalProperties": False}
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def all(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def for_role(self, role: Role) -> List[ToolSpec]:
        """仅返回该角色可见（allowed_roles 包含）的工具，供 LLM 决策子集。
        📌 技术点8/4：按角色裁剪工具子集，学员看不到数据查询/通知等越权工具。"""
        return [t for t in self._tools.values() if role in t.allowed_roles]

    def openai_tools(self, role: Role) -> List[dict]:
        """OpenAI 兼容 tools 结构（name/description/parameters），喂给 LLM 做 Function Calling。
        📌 role **必传**：早期版本这里返回 self._tools.values() 全量工具，等于绕过 for_role() 的
        RBAC 裁剪——一旦被误用，学员角色的决策里也会出现教师工具（执行层虽有二次校验，
        但"看不见"才是第一道防线）。故改为强制按角色裁剪，调用方必须显式声明是谁在问。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.for_role(role)
        ]


registry = ToolRegistry()
