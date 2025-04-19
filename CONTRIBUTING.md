# Contributing to plexe

Thank you for considering contributing to plexe! Your contributions help improve this project for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Development Setup](#development-setup)
- [Style Guides](#style-guides)
  - [Coding Standards](#coding-standards)
  - [Commit Messages](#commit-messages)

## Code of Conduct

By participating in this project, you agree to uphold our [Code of Conduct](link-to-code-of-conduct), which outlines expectations for respectful and inclusive interactions.

## How Can I Contribute?

### Reporting Bugs

If you encounter a bug, please:

1. **Search Existing Issues**: Check if the issue has already been reported.
2. **Open a New Issue**: If not found, create a new issue and include:
   - A descriptive title.
   - Steps to reproduce the bug.
   - Expected and actual behavior.
   - Screenshots or code snippets, if applicable.

### Suggesting Enhancements

To propose new features or improvements:

1. **Search Existing Issues**: Ensure the suggestion hasn't been made.
2. **Open a New Issue**: Provide:
   - A clear description of the enhancement.
   - Rationale for the suggestion.
   - Any relevant examples or references.

### Submitting Pull Requests

For code contributions:

1. **Fork the Repository**: Create your own copy of the repo.
2. **Create a Branch**: Use a descriptive name (e.g., `feature/new-model` or `bugfix/issue-123`).
3. **Make Changes**: Implement your changes with clear and concise code.
4. **Write Tests**: Ensure new features or bug fixes are covered by tests.
5. **Commit Changes**: Follow our commit message guidelines.
6. **Push to Your Fork**: Upload your changes.
7. **Open a Pull Request**: Provide a detailed description of your changes and reference any related issues.

## Development Setup

To set up the development environment:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/plexe-ai/plexe.git
    cd plexe
    ```

2. **Install Dependencies**:

    ```bash
    python setup.py
    ```

3. **Run Tests**:

    ```bash
    pytest
    ```

Ensure all tests pass before making contributions.

## Style Guides

### Coding Standards

Adhere to PEP 8 guidelines for Python code. Key points include:

- Use 4 spaces per indentation level.
- Limit lines to 79 characters.
- Use meaningful variable and function names.
- Include docstrings for all public modules, classes, and functions.

### Commit Messages

Write clear and concise commit messages:

- **Format**: `<type>(<scope>): <subject>`
  - **Type**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
  - **Scope**: Optional, e.g., `data`, `model`
  - **Subject**: Brief description (max 50 characters)

- **Example**:

    ```bash
    feat(model): add support for gemini
    ```
