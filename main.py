import streamlit as st
import sqlite3
import subprocess

# Create a connection to the SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create table for users if not exists
c.execute('''CREATE TABLE IF NOT EXISTS users (
             username TEXT PRIMARY KEY,
             password TEXT)''')
conn.commit()

def create_user(username, password):
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()

def login(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    if c.fetchone():
        return True
    else:
        return False

def open_next_page():

---------------------------------------------- if yot have to ue code contact me freely ---------------------------------------------------------------

if __name__ == "__main__":
    main()