from app.db import engine, Base
from app.models import User, Movie, Interaction, EvaluationMetric

def init_db():
    print("Initializing database tables...")
    # Base.metadata.create_all binds our model classes to the engine schema 
    Base.metadata.create_all(bind=engine)
    print("Database setup complete! 'recommender.db' file created successfully.")

if __name__ == "__main__":
    init_db()