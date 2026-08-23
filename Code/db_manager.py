from psycopg2 import DATETIME
import os
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables relative to this file's directory
db_manager_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(db_manager_dir, ".env"))
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

def get_connection(database_name=None):
    """
    Establish an active connection to the database. Use for data updating.
    Must close the connection after calling this function.
    """
    db = database_name if database_name is not None else DB_NAME
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=db,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL database '{db}': {e}")
        raise e

def get_engine():
    """
    Establish a lazy engine connection to the database. Use for data modeling. 
    This function is slower than get_connection() but automatically manages resource clean-up.
    """
    return create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

def create_database_if_not_exists():
    """
    Connect to the database specified in .env or create the database if it doesn't exist.
    """
    try:
        # Connect to database
        conn = get_connection(database_name=DB_NAME)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Database '{DB_NAME}' does not exist. Creating...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Could not check/create database '{DB_NAME}' automatically: {e}")
        print("Will proceed assuming database already exists.")

def init_tables():
    """
    Initialize all required database tables.
    """
    create_database_if_not_exists()
    conn = get_connection()
    cursor = conn.cursor()

    # Technology-related news headlines table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tech_headlines (
            timestamp_utc TIMESTAMPTZ UNIQUE,
            headline TEXT,
            threats_and_vulnerabilities FLOAT,
            opportunities_for_growth FLOAT,
            neutral FLOAT,
            news_score FLOAT
        );
    """)
    
    # Datadog stock returns table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ddog_returns (
            timestamp_utc TIMESTAMPTZ UNIQUE,
            close FLOAT,
            return FLOAT
        );
    """)

    # PYPL stock returns table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pypl_returns (
            timestamp_utc TIMESTAMPTZ UNIQUE,
            close FLOAT,
            return FLOAT
        );
    """)

    # Datadog announcements table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS announcements (
            timestamp_utc TIMESTAMPTZ UNIQUE,
            headline TEXT,
            sentiment_score FLOAT
        );
    """)

    # DDOG quarterly metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qtrly_metrics (
            timestamp_utc TIMESTAMPTZ,
            metric TEXT,
            value FLOAT,
            UNIQUE (timestamp_utc, metric)
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Database tables initialized successfully.")

if __name__ == "__main__":
    init_tables()
