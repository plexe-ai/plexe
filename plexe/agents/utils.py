"""
Shared utilities for agents.

Helper functions used across multiple agent implementations.
"""

from __future__ import annotations


def format_user_feedback_for_prompt(user_feedback: dict | str | None) -> str:
    """
    Format user feedback for inclusion in agent prompts.

    Extracts feedback text from various formats and formats it for LLM consumption.
    Agents should incorporate this feedback into their decision-making.

    Args:
        user_feedback: Feedback from context.scratch["_user_feedback"]
            Can be:
            - None: No feedback provided
            - str: Direct feedback text
            - dict: Structured feedback with 'comments', 'feedback', or 'requested_changes' fields

    Returns:
        Formatted string for prompt injection, or empty string if no feedback

    Examples:
        >>> format_user_feedback_for_prompt(None)
        ''

        >>> format_user_feedback_for_prompt("Focus on temporal features")
        '\\n## ⚠️ USER FEEDBACK:\\nThe user has provided guidance...\\nFocus on temporal features\\n...'

        >>> format_user_feedback_for_prompt({"comments": "Try neural networks", "approved": True})
        '\\n## ⚠️ USER FEEDBACK:\\nThe user has provided guidance...\\nTry neural networks\\n...'
    """
    if not user_feedback:
        return ""

    # Handle different feedback formats
    if isinstance(user_feedback, str):
        feedback_text = user_feedback
    elif isinstance(user_feedback, dict):
        # Extract relevant fields from structured feedback
        parts = []

        # Primary field: comments (standard field from FeedbackSubmission)
        if "comments" in user_feedback and user_feedback["comments"]:
            parts.append(user_feedback["comments"])

        # Fallback field: feedback (legacy/alternative field)
        if "feedback" in user_feedback and user_feedback["feedback"]:
            parts.append(user_feedback["feedback"])

        # Additional field: requested_changes (list of change requests)
        if "requested_changes" in user_feedback and user_feedback["requested_changes"]:
            changes = user_feedback["requested_changes"]
            if isinstance(changes, list) and changes:
                changes_text = "\n".join([f"- {change}" for change in changes])
                parts.append(f"Requested changes:\n{changes_text}")

        feedback_text = "\n\n".join(parts)
    else:
        return ""

    if not feedback_text.strip():
        return ""

    return (
        "\n"
        "## ⚠️ USER FEEDBACK:\n"
        "The user has provided guidance that you MUST incorporate into your work:\n"
        "\n"
        f"{feedback_text}\n"
        "\n"
        "**IMPORTANT**: Adjust your approach based on this feedback. The user's domain knowledge "
        "should override default assumptions and guide your decisions.\n"
    )
