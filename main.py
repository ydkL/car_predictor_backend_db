from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String, Float, func, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List

from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Path to SQLite database file
DATABASE_FILE = "test.db"

# Create SQLAlchemy engine
engine = create_engine(f"sqlite:///{DATABASE_FILE}")

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()

# Define a simple ORM model
class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    modelName = Column(String)
    segmentName = Column(String)
    Grid = Column(Float)
    Comfort = Column(Float)
    Tech = Column(Float)
    Visualize = Column(Float)
    Volume = Column(Float)
    Reliability = Column(Float)
    Security = Column(Float)
    Service = Column(Float)
    Insulation = Column(Float)

# Create tables in the database
Base.metadata.create_all(bind=engine)

# Pydantic model for item creation
class ItemCreate(BaseModel):
    modelName: str
    segmentName: str
    Grid: float
    Comfort: float
    Tech: float
    Visualize: float
    Volume: float
    Reliability: float
    Security: float
    Service: float
    Insulation: float
    
# Pydantic model for response
class ModelStats(BaseModel):
    modelName: str
    Grid: float
    Comfort: float
    Tech: float
    Visualize: float
    Volume: float
    Reliability: float
    Security: float
    Service: float
    Insulation: float

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Define SQLAlchemy database engine asynchronously
DATABASE_URL = "sqlite+aiosqlite:///./test.db"
async_engine = create_async_engine(DATABASE_URL, echo=True)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint to create an item
@app.post("/items/")
async def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    print('Gotted')
    db_item = Item(**item.dict())
    print(db_item)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

# Endpoint to get an item by ID
@app.get("/items/{item_id}")
async def read_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(Item).filter(Item.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


# Function to get model statistics asynchronously
async def get_model_stats():
    async with async_engine.connect() as connection:
        # Define the columns and their average values
        columns = [
          func.avg(Item.Grid).label("Grid"),
            func.avg(Item.Comfort).label("Comfort"),
            func.avg(Item.Tech).label("Tech"),
            func.avg(Item.Visualize).label("Visualize"),
            func.avg(Item.Volume).label("Volume"),
            func.avg(Item.Reliability).label("Reliability"),
            func.avg(Item.Security).label("Security"),
            func.avg(Item.Service).label("Service"),
            func.avg(Item.Insulation).label("Insulation")
        ]

        # Query to get average values grouped by model name
        stmt = select(Item.modelName, *columns).group_by(Item.modelName)
        result = await connection.execute(stmt)
        models = result.fetchall()

        model_stats = []
        for model in models:
            model_stats.append(
                ModelStats(
                    modelName=model[0],
                    Grid=model[1],
                    Comfort=model[2],
                    Tech=model[3],
                    Visualize=model[4],
                    Volume=model[5],
                    Reliability=model[6],
                    Security=model[7],
                    Service=model[8],
                    Insulation=model[9]
                )
            )
        
        return model_stats

# Endpoint to get all models with average values grouped by model name
@app.get("/model-stats/", response_model=List[ModelStats])
async def get_model_stats_endpoint():
    return await get_model_stats()