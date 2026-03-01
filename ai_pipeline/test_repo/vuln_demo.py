import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("app.db")
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    return conn.execute(query).fetchone()
