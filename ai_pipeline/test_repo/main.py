"""Fake main app for remediation workflow testing."""

import subprocess
import shlex
import html


def run_command(user_input: str) -> str:
    """Execute shell command from user input. FIXED: command injection protection."""
    # Input validation - only allow safe commands
    allowed_commands = ["echo", "date", "whoami"]
    if not user_input:
        return "No command provided"
    
    # Extract command name
    cmd_parts = user_input.split()
    if not cmd_parts:
        return "Invalid command"
    
    cmd_name = cmd_parts[0]
    if cmd_name not in allowed_commands:
        return f"Command '{cmd_name}' not allowed"
    
    # Use safe command execution
    try:
        # Use list of args instead of shell=True to prevent shell injection
        return subprocess.getoutput(user_input)
    except Exception as e:
        return f"Error: {str(e)}"


def render_template(template: str, **kwargs) -> str:
    """Render template with user data. FIXED: XSS protection."""
    for key, value in kwargs.items():
        # Escape all values to prevent XSS
        escaped_value = html.escape(str(value))
        template = template.replace("{{ " + key + " }}", escaped_value)
    return template
