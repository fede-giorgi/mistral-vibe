"""Fake main app for remediation workflow testing."""

import subprocess


def run_command(user_input: str) -> str:
    """Execute shell command from user input. FAKE: command injection."""
    return subprocess.getoutput(user_input)


def render_template(template: str, **kwargs) -> str:
    """Render template with user data. FAKE: no escaping."""
    for key, value in kwargs.items():
        template = template.replace("{{ " + key + " }}", str(value))
    return template
