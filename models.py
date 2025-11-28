import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Define the base class for declarative models
Base = declarative_base()

# Define the database engine for SQLite. `check_same_thread` is needed for SQLite with Flask.
DATABASE_URL = "sqlite:///database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define the Event model
class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    event_type = Column(String(50), nullable=False)
    zone_name = Column(String(50), nullable=False)
    screenshot_path = Column(String(255), nullable=True)
    # Optional tracking metadata
    person_slot = Column(Integer, nullable=True)  # 1..5
    person_label = Column(String(100), nullable=True)

    def to_dict(self):
        """Converts the event object to a dictionary for JSON serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'zone_name': self.zone_name,
            'screenshot_path': self.screenshot_path,
            'person_slot': self.person_slot,
            'person_label': self.person_label,
        }

# Function to create the database and tables
def init_db():
    """Initializes the database and creates tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    # Lightweight migration: add missing columns to existing table if needed
    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(events);"))}
        if 'person_slot' not in cols:
            conn.execute(text("ALTER TABLE events ADD COLUMN person_slot INTEGER;"))
        if 'person_label' not in cols:
            conn.execute(text("ALTER TABLE events ADD COLUMN person_label VARCHAR(100);"))

if __name__ == "__main__":
    # Allows creating the database from the command line by running: python models.py
    print("Initializing database...")
    init_db()
    print("Database initialized.")
