import pytest

from app.tools.calculator import CalculatorTool

tool = CalculatorTool()


class TestBasicArithmetic:
    async def test_addition(self):
        assert await tool.execute(expression="2 + 3") == 5

    async def test_precedence_and_parentheses(self):
        assert await tool.execute(expression="(2 + 3) * 4") == 20

    async def test_power(self):
        assert await tool.execute(expression="2 ** 10") == 1024

    async def test_floor_div_and_mod(self):
        assert await tool.execute(expression="7 // 2") == 3
        assert await tool.execute(expression="7 % 2") == 1

    async def test_unary_minus(self):
        assert await tool.execute(expression="-5 + 3") == -2

    async def test_float_result(self):
        assert await tool.execute(expression="1 / 4") == 0.25


class TestErrorHandling:
    async def test_division_by_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="division by zero"):
            await tool.execute(expression="1 / 0")

    async def test_invalid_syntax_raises_value_error(self):
        with pytest.raises(ValueError, match="invalid expression syntax"):
            await tool.execute(expression="2 + * 3")

    async def test_empty_expression_raises(self):
        with pytest.raises(ValueError):
            await tool.execute(expression="")


class TestInjectionIsRejectedNotExecuted:
    """The whole point of this tool: it must be impossible to reach
    arbitrary code execution through the expression string.
    """

    async def test_function_call_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported expression element"):
            await tool.execute(expression="__import__('os').system('echo pwned')")

    async def test_name_reference_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported expression element"):
            await tool.execute(expression="os")

    async def test_attribute_access_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported expression element"):
            await tool.execute(expression="(1).__class__")

    async def test_list_literal_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported expression element"):
            await tool.execute(expression="[1, 2, 3]")

    async def test_string_literal_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported constant"):
            await tool.execute(expression="'hello'")

    async def test_comparison_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported expression element"):
            await tool.execute(expression="1 == 1")

    async def test_boolean_constant_is_rejected(self):
        with pytest.raises(ValueError, match="unsupported constant"):
            await tool.execute(expression="True")


class TestInputSchema:
    def test_requires_expression_field(self):
        from pydantic import ValidationError

        from app.tools.calculator import CalculatorInput

        with pytest.raises(ValidationError):
            CalculatorInput()
