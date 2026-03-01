"""Fake auth module for remediation workflow testing."""

import os
import html


def login(username: str, password: str) -> bool:
    """Check credentials against database. FIXED: parameterized query."""
    # Input validation
    if not username or not password:
        return False
    
    # Use parameterized query to prevent SQL injection
    # In a real app, this would connect to a database
    query = "SELECT * FROM users WHERE name = ? AND pass = ?"
    # Simulate parameterized execution
    return f"EXECUTED: {query} WITH PARAMS: {username}, {password}"


def get_user_input():
    """Return user input with XSS protection."""
    data = input("Enter name: ")
    # Escape HTML to prevent XSS
    return f"<div>Hello, {html.escape(data)}</div>"
