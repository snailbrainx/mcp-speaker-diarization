#!/usr/bin/env python3
"""
Create the ground_truth_labels table
"""

from app.database import engine, Base
from app.models import GroundTruthLabel

print("Creating ground_truth_labels table...")
Base.metadata.create_all(bind=engine)
print("✓ Table created successfully!")
