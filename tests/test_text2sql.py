# -*- coding: utf-8 -*-
import pytest

from app import data


def test_text2sql_eval_5_5():
    """Text2SQL 评测集：5/5 命中 + 5/5 真执行（简历/备考库口径；2026-09-01 增考勤用例）。"""
    res = data.eval_text2sql()
    assert res["total"] == 5
    assert res["hit"] == 5
    assert res["executed"] == 5
    assert res["rate"] == "100%"
    assert res["exec_rate"] == "100%"


def test_rotation_feasibility_runs():
    out = data.check_rotation_feasibility()
    assert "轮转分配可行性检测" in out


# ============ 2026-09-08 code review P0-5 修复：AST 解析防字面量绕过 ============
class TestAstCheckTenant:
    """验证 _ast_check_tenant 拦截字符串层漏检的绕过。"""

    def test_正常_tenant_id_等值占位_通过(self):
        sql = "SELECT * FROM rotation WHERE tenant_id = ? AND id = 1"
        assert data._ast_check_tenant(sql) is None

    def test_占位在左_tenant_id_在右_也通过(self):
        """? = tenant_id 形态也算（虽然实际少用，但 SQL 等价）"""
        sql = "SELECT * FROM rotation WHERE ? = tenant_id"
        assert data._ast_check_tenant(sql) is None

    def test_字面量_tenant_id_被拦截(self):
        """P0 关键场景：'tenant_id' = 'x' 含子串但非列引用 → 字符串层漏检，AST 拦"""
        # 注意：SQL 文本里确实有 "tenant_id" 字样，但作为字符串字面量，不是列
        sql = "SELECT * FROM rotation WHERE 'tenant_id' = 'x'"
        err = data._ast_check_tenant(sql)
        assert err is not None
        assert "tenant_id" in err

    def test_子查询_被拦截(self):
        """P0 关键场景：tenant_id = (SELECT ...) 子查询绕过"""
        sql = "SELECT * FROM rotation WHERE tenant_id = (SELECT tenant_id FROM other_table)"
        err = data._ast_check_tenant(sql)
        assert err is not None
        assert "子查询" in err

    def test_多语句_被拦截(self):
        """P0 关键场景：'SELECT 1; DROP TABLE users' 拼接"""
        sql = "SELECT 1; DROP TABLE users"
        err = data._ast_check_tenant(sql)
        assert err is not None
        assert "多语句" in err

    def test_无_where_被拦截(self):
        sql = "SELECT * FROM rotation"
        err = data._ast_check_tenant(sql)
        assert err is not None

    def test_非_select_被拦截(self):
        sql = "DELETE FROM rotation WHERE tenant_id = ?"
        err = data._ast_check_tenant(sql)
        assert err is not None
        assert "SELECT" in err

    def test_占位是字面量_被拦截(self):
        """P0 关键场景：tenant_id = 'literal' 字面量绕过"""
        sql = "SELECT * FROM rotation WHERE tenant_id = 'capital-medical'"
        # 字符串层会先拦（re.search 检查 "tenant_id = '..."），AST 层也会拦
        with pytest.raises(ValueError, match="禁止写死租户"):
            data.assert_sql_safe(sql, "capital-medical")
