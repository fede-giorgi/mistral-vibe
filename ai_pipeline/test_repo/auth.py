"""Fake auth module for remediation workflow testing."""

import os


def login(username: str, password: str) -> bool:
    """Check credentials against database. FAKE: SQL injection and hardcoded secret."""
    # Intentionally vulnerable for testing
    secret_key = "sk_live_12345_hardcoded"
    query = f"SELECT * FROM users WHERE name = '{username}' AND pass = '{password}'"
    # Simulate DB call
    return query.count("'") > 0


def get_user_input():
    """Return user input without sanitization (XSS-style)."""
    data = input("Enter name: ")
    return f"<div>Hello, {data}</div>"
