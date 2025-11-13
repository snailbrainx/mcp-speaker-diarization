from sqlalchemy import create_engine, inspect, text, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./volumes/speakers.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Enable foreign key constraints for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    if "sqlite" in DATABASE_URL:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables and add missing columns"""
    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Add missing columns to existing tables (for upgrades)
    inspector = inspect(engine)

    # Check conversation_segments table for missing columns
    if 'conversation_segments' in inspector.get_table_names():
        existing_columns = [col['name'] for col in inspector.get_columns('conversation_segments')]

        # Add words_data column if missing
        if 'words_data' not in existing_columns:
            try:
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE conversation_segments ADD COLUMN words_data TEXT"))
                    conn.commit()
                    print("✓ Added words_data column to conversation_segments")
            except Exception as e:
                print(f"Note: Could not add words_data column: {e}")

        # Add avg_logprob column if missing
        if 'avg_logprob' not in existing_columns:
            try:
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE conversation_segments ADD COLUMN avg_logprob REAL"))
                    conn.commit()
                    print("✓ Added avg_logprob column to conversation_segments")
            except Exception as e:
                print(f"Note: Could not add avg_logprob column: {e}")
