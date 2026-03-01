import sqlite3
import html

def get_user(username, password):
    # Input validation
    if not username or not password:
        return None
    
    conn = sqlite3.connect("app.db")
    query = "SELECT * FROM users WHERE username = ? AND password = ?"
    cursor = conn.cursor()
    cursor.execute(query, (username, password))
    result = cursor.fetchone()
    conn.close()
    return result

def render_user_data(user_data):
    """Render user data with XSS protection."""
    if not user_data:
        return "No user data"
    
    # Escape all string fields
    escaped_data = {}
    for key, value in user_data.items():
        if isinstance(value, str):
            escaped_data[key] = html.escape(value)
        else:
            escaped_data[key] = value
    
    return escaped_data
