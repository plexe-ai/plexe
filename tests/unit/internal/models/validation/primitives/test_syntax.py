import pytest
from plexe.internal.models.validation.primitives.syntax import SyntaxValidator


@pytest.fixture
def syntax_validator():
    """
    Fixture to provide an instance of SyntaxValidator.

    :return: An instance of SyntaxValidator.
    """
    return SyntaxValidator()


def test_valid_code(syntax_validator):
    """Test that the validate method correctly identifies valid Python code."""
    valid_code = "def add(a, b):\n" "    return a + b\n\n" "result = add(2, 3)\n"
    result = syntax_validator.validate(valid_code)
    assert result.passed is True
    assert result.message == "Syntax is valid."
    assert result.exception is None


def test_invalid_code(syntax_validator):
    """Test that the validate method correctly identifies invalid Python code."""
    invalid_code = "def add(a, b):\n" "    return a + b\n" "result = add(2, 3\n"  # Missing closing parenthesis
    result = syntax_validator.validate(invalid_code)
    assert result.passed is False
    assert "Syntax is not valid" in result.message
    assert "line 3" in result.message  # Ensures the line number is reported
    assert result.exception is not None
    assert isinstance(result.exception, SyntaxError)


def test_empty_code(syntax_validator):
    """Test that the validate method handles empty code correctly."""
    empty_code = ""
    result = syntax_validator.validate(empty_code)
    assert result.passed is True
    assert result.message == "Syntax is valid."
    assert result.exception is None


def test_code_with_comments(syntax_validator):
    """Test that the validate method handles code containing only comments."""
    comment_code = "# This is a comment\n" "# Another comment line\n"
    result = syntax_validator.validate(comment_code)
    assert result.passed is True
    assert result.message == "Syntax is valid."
    assert result.exception is None


def test_code_with_syntax_warning(syntax_validator):
    """Test that the validate method handles code with a warning but no syntax error."""
    warning_code = "x = 1  # Variable assigned but not used"
    result = syntax_validator.validate(warning_code)
    assert result.passed is True
    assert result.message == "Syntax is valid."
    assert result.exception is None


def test_code_with_non_ascii_characters(syntax_validator):
    """Test that the validate method handles code with non-ASCII characters."""
    non_ascii_code = "def greet():\n" '   return "Hello, \u4f60\u597d!"'  # Includes Chinese characters for "Hello"
    result = syntax_validator.validate(non_ascii_code)
    assert result.passed is True
    assert result.message == "Syntax is valid."
    assert result.exception is None


def test_code_with_indentation_error(syntax_validator):
    """Test that the validate method correctly identifies indentation errors."""
    indentation_error_code = "def add(a, b):\n" "return a + b"  # Missing indentation for the return statement
    result = syntax_validator.validate(indentation_error_code)
    assert result.passed is False
    assert "Syntax is not valid" in result.message
    assert "line 2" in result.message  # Ensures the line number is reported
    assert result.exception is not None
    assert isinstance(result.exception, SyntaxError)


def test_code_with_nested_functions(syntax_validator):
    """Test that the validate method handles code with nested functions."""
    nested_function_code = "def outer():\n" "    def inner():\n" '        return "Hello"\n' "    return inner()\n"
    result = syntax_validator.validate(nested_function_code)
    assert result.passed is True
    assert result.message == "Syntax is valid."
    assert result.exception is None


def test_code_with_large_input(syntax_validator):
    """Test that the validate method handles a large amount of valid code."""
    large_code = "\n".join([f"def func{i}():\n    return {i}" for i in range(1000)])
    result = syntax_validator.validate(large_code)
    assert result.passed is True
    assert result.message == "Syntax is valid."
    assert result.exception is None
