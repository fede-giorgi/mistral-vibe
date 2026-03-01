import sqlite3

def get_user(username, password):
    conn = sqlite3.connect("app.db")
    query = "SELECT * FROM users WHERE username = ? AND password = ?"
    cursor = conn.cursor()
    cursor.execute(query, (username, password))
    result = cursor.fetchone()
    conn.close()
    return result
