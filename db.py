import sqlite3

def save_to_db(df):
    conn = sqlite3.connect("solar.db")
    df.to_sql("solar_data", conn, if_exists="replace", index=False)
    conn.close()