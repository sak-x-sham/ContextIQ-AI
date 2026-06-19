import sqlite3

DB_NAME = "chat_history.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        chat_id TEXT PRIMARY KEY,
        chat_name TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        role TEXT,
        content TEXT
    )
    """)

    conn.commit()
    conn.close()

def create_chat(chat_id, chat_name):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO chats VALUES (?, ?)",
        (chat_id, chat_name)
    )

    conn.commit()
    conn.close()

def save_message(chat_id, role, content):

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO messages
        (chat_id, role, content)
        VALUES (?, ?, ?)
        """,
        (chat_id, role, content)
    )

    conn.commit()
    conn.close()

def load_chats():

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT chat_id, chat_name
        FROM chats
        """
    )

    rows = cur.fetchall()

    conn.close()

    return rows

def load_messages(chat_id):

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT role, content
        FROM messages
        WHERE chat_id=?
        ORDER BY id
        """,
        (chat_id,)
    )

    rows = cur.fetchall()

    conn.close()

    return [
        {
            "role": r,
            "content": c
        }
        for r, c in rows
    ]

def delete_chat(chat_id):

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM chats WHERE chat_id=?",
        (chat_id,)
    )

    cur.execute(
        "DELETE FROM messages WHERE chat_id=?",
        (chat_id,)
    )

    conn.commit()
    conn.close()



def clear_chat_messages(chat_id):

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        DELETE FROM messages
        WHERE chat_id = ?
        """,
        (chat_id,)
    )

    conn.commit()
    conn.close()

def delete_all_chats():

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("DELETE FROM messages")
    cur.execute("DELETE FROM chats")

    conn.commit()
    conn.close()

def rename_chat(chat_id, new_name):

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE chats
        SET chat_name = ?
        WHERE chat_id = ?
        """,
        (new_name, chat_id)
    )

    conn.commit()
    conn.close()