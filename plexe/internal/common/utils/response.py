import json
import re
import logging
import pandas as pd

import black

logging.getLogger("blib2to3.pgen2.driver").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    # Sometimes chatgpt-turbo forget the last curly bracket, so we try to add it back when no json is found
    if len(json_objects) == 0 and not text.endswith("}"):
        json_objects = extract_jsons(text + "}")
        if len(json_objects) > 0:
            return json_objects

    return json_objects


def trim_long_string(string, threshold=5100, k=2500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string


def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [format_code(c) for c in parsed_codes if is_valid_python_script(c)]
    return format_code("\n\n".join(valid_code_blocks))


def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def extract_performance(output: str) -> float | None:
    """Extract the performance metric from the output."""
    try:
        last_line = output.strip().split("\n")[-1]

        # Looking for format "MetricName: value"
        if ":" not in last_line:
            raise RuntimeError("No colon found in last line")

        value_str = last_line.split(":")[-1].strip()

        try:
            value = float(value_str)
            logger.debug(f"Successfully parsed value: {value}")
            return value
        except ValueError as e:
            raise RuntimeError(f"Could not convert '{value_str}' to float: {e}") from e

    except Exception as e:
        raise RuntimeError(f"Error extracting run performance: {e}") from e


def extract_json_array(text: str) -> str:
    """
    Extract a JSON array from an LLM response, handling common formatting issues.

    This function cleans up LLM responses that may contain JSON arrays embedded in
    markdown code blocks, additional explanatory text, or other formatting artifacts.

    Args:
        text: The raw text output from an LLM

    Returns:
        A cleaned JSON array string ready for parsing
    """
    cleaned_text = text

    # Remove code block markers if present
    if "```" in cleaned_text:
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.DOTALL)
        if json_match:
            cleaned_text = json_match.group(1)

    # Remove any language tags or backticks
    cleaned_text = cleaned_text.replace("json", "").replace("`", "").strip()

    # Find the actual JSON array if needed
    start_idx = cleaned_text.find("[")
    end_idx = cleaned_text.rfind("]")
    if start_idx >= 0 and end_idx >= 0:
        cleaned_text = cleaned_text[start_idx : end_idx + 1]

    return cleaned_text


def json_to_dataframe(text: str) -> "pd.DataFrame":
    """
    Convert LLM-generated JSON text to a pandas DataFrame.

    This function handles common errors in LLM responses when creating DataFrames:
    1. Extracts and cleans the JSON array from text
    2. Validates it's a proper array structure
    3. Creates a DataFrame with appropriate error handling

    Args:
        text: Raw text output from an LLM that should contain a JSON array

    Returns:
        pandas DataFrame created from the JSON data

    Raises:
        ValueError: If the response can't be parsed as a valid JSON array
    """
    import pandas as pd

    # Extract the JSON array text
    json_text = extract_json_array(text)

    try:
        # Parse the JSON
        data = json.loads(json_text)

        # Handle both single object and array of objects
        if not isinstance(data, list):
            # If it's a single object, convert it to a list with one item
            if isinstance(data, dict):
                logger.warning("JSON is a single object, converting to list")
                data = [data]
            else:
                raise ValueError(f"JSON is not an array or object: {json_text[:100]}...")

        # Check if it's empty
        if len(data) == 0:
            logger.warning("JSON array is empty")
            # Return empty DataFrame
            return pd.DataFrame()

        # Create DataFrame from the parsed JSON array
        return pd.DataFrame(data)

    except json.JSONDecodeError as e:
        # Log details about the error
        logger.error(f"Failed to parse JSON: {str(e)}")
        logger.debug(f"Attempted to parse: {json_text[:200]}...")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Error converting JSON to DataFrame: {str(e)}")
        raise ValueError(f"Error converting JSON to DataFrame: {str(e)}")
