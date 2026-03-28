"""
Unit tests for hard_rule_check — a pure function in validator.py.

Note: validator.py has module-level LLM initialization. The sys.modules mocks in
conftest.py prevent those from running, allowing hard_rule_check to be imported
and tested in isolation without any API keys.
"""
import pytest
from pipeline.nodes.validator import hard_rule_check


SAFE_QUERIES = [
    "SELECT COUNT(*) FROM orders",
    "SELECT city, COUNT(*) FROM orders GROUP BY city",
    "SELECT u.city, AVG(o.grand_total) FROM orders o JOIN users u ON o.user_id = u.id GROUP BY u.city",
    "WITH revenue AS (SELECT SUM(total_price) AS rev FROM order_items) SELECT * FROM revenue",
    "SELECT * FROM products WHERE category_id = 1 ORDER BY name LIMIT 10",
    "SELECT id, name FROM users WHERE status = 'active'",
]

BANNED_QUERIES = [
    ("DELETE FROM orders WHERE id = 1", "DELETE"),
    ("DROP TABLE orders", "DROP"),
    ("INSERT INTO orders VALUES (1, 2, 3)", "INSERT"),
    ("UPDATE orders SET status = 'cancelled' WHERE id = 5", "UPDATE"),
    ("TRUNCATE TABLE order_items", "TRUNCATE"),
    ("GRANT ALL ON orders TO public", "GRANT"),
    ("ALTER TABLE orders ADD COLUMN foo TEXT", "ALTER"),
]


class TestHardRuleCheckSafeQueries:
    @pytest.mark.parametrize("sql", SAFE_QUERIES)
    def test_safe_sql_returns_safe_status(self, sql):
        result = hard_rule_check(sql)
        assert result["status"] == "safe", f"Expected 'safe' for: {sql}"

    @pytest.mark.parametrize("sql", SAFE_QUERIES)
    def test_safe_sql_returns_empty_repair_hint(self, sql):
        result = hard_rule_check(sql)
        assert result["repair_hint"] == ""

    def test_lowercase_safe_sql_passes(self):
        result = hard_rule_check("select count(*) from orders")
        assert result["status"] == "safe"

    def test_mixed_case_safe_sql_passes(self):
        result = hard_rule_check("Select Count(*) From Orders")
        assert result["status"] == "safe"


class TestHardRuleCheckBannedKeywords:
    @pytest.mark.parametrize("sql,keyword", BANNED_QUERIES)
    def test_banned_keyword_returns_invalid_status(self, sql, keyword):
        result = hard_rule_check(sql)
        assert result["status"] == "invalid", f"Expected 'invalid' for keyword {keyword} in: {sql}"

    @pytest.mark.parametrize("sql,keyword", BANNED_QUERIES)
    def test_banned_keyword_name_appears_in_reason(self, sql, keyword):
        result = hard_rule_check(sql)
        assert keyword in result["reason"]

    @pytest.mark.parametrize("sql,keyword", BANNED_QUERIES)
    def test_banned_keyword_has_non_empty_repair_hint(self, sql, keyword):
        result = hard_rule_check(sql)
        assert result["repair_hint"] != ""

    def test_lowercase_banned_keyword_is_caught(self):
        result = hard_rule_check("delete from users where id = 1")
        assert result["status"] == "invalid"

    def test_mixed_case_banned_keyword_is_caught(self):
        result = hard_rule_check("Delete From users Where id = 1")
        assert result["status"] == "invalid"


class TestHardRuleCheckBoundary:
    def test_keyword_inside_word_is_not_flagged(self):
        # "delete" as part of a column/table name — split() won't isolate it
        result = hard_rule_check("SELECT soft_delete_flag FROM orders")
        assert result["status"] == "safe"

    def test_returns_dict_with_required_keys(self):
        result = hard_rule_check("SELECT 1")
        assert "status" in result
        assert "reason" in result
        assert "repair_hint" in result

    def test_only_first_banned_keyword_triggers(self):
        # Multiple banned keywords — should fail on first match
        result = hard_rule_check("DELETE FROM orders; DROP TABLE users")
        assert result["status"] == "invalid"
