# import sqlite3
# import os

# os.makedirs("database", exist_ok=True)

# conn = sqlite3.connect("database/lungvision.db")
# cursor = conn.cursor()

# # ---------------- USERS ----------------

# cursor.execute("""
# CREATE TABLE IF NOT EXISTS users(
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     full_name TEXT NOT NULL,
#     email TEXT UNIQUE NOT NULL,
#     phone TEXT,
#     password_hash TEXT NOT NULL,
#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# )
# """)

# cursor.execute("""
# CREATE TABLE IF NOT EXISTS predictions (

#     id INTEGER PRIMARY KEY AUTOINCREMENT,

#     user_id INTEGER,

#     image_name TEXT,

#     disease TEXT,

#     confidence REAL,

#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

#     FOREIGN KEY(user_id) REFERENCES users(id)

# )
# """)

# # ---------------- PREDICTIONS ----------------

# cursor.execute("""
# CREATE TABLE IF NOT EXISTS predictions(
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     user_id INTEGER,
#     image_name TEXT,
#     predicted_disease TEXT,
#     confidence REAL,
#     prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

#     FOREIGN KEY(user_id)
#     REFERENCES users(id)
# )
# """)

# # ---------------- APPOINTMENTS ----------------

# cursor.execute("""
# CREATE TABLE IF NOT EXISTS appointments(
#     id INTEGER PRIMARY KEY AUTOINCREMENT,

#     user_id INTEGER,

#     patient_name TEXT,

#     phone TEXT,

#     disease TEXT,

#     doctor TEXT,

#     appointment_date DATE,

#     appointment_time TIME,

#     symptoms TEXT,

#     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

#     FOREIGN KEY(user_id)
#     REFERENCES users(id)
# )
# """)

# conn.commit()
# conn.close()

# print("Database Created Successfully!")

import sqlite3
import os

os.makedirs("database", exist_ok=True)

conn = sqlite3.connect("database/lungvision.db")
cursor = conn.cursor()

# ---------------- USERS ----------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# ---------------- PREDICTIONS ----------------
# (Only ONE definition now — matches the columns app.py actually inserts into)

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    image_name TEXT,
    predicted_disease TEXT,
    confidence REAL,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(user_id)
    REFERENCES users(id)
)
""")

# ---------------- APPOINTMENTS ----------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS appointments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    patient_name TEXT,
    phone TEXT,
    disease TEXT,
    doctor TEXT,
    appointment_date DATE,
    appointment_time TIME,
    symptoms TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(user_id)
    REFERENCES users(id)
)
""")

# ---------------- CONTACT MESSAGES ----------------
# (This was missing entirely — needed for /submit_contact in app.py)

cursor.execute("""
CREATE TABLE IF NOT EXISTS contact_messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT,
    email TEXT,
    phone TEXT,
    subject TEXT,
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Database Created Successfully!")