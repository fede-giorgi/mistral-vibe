import sqlite3
from flask import Flask, request

app = Flask(__name__)

@app.route("/search")
def search():
    query = request.args.get("q", "")
    db = sqlite3.connect("app.db")
    results = db.execute(
        f"SELECT * FROM users WHERE name LIKE '%{query}%'"
    ).fetchall()
    return str(results)
