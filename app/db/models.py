# db/models.py
from sqlalchemy import (
    Column, String, Integer, DateTime, Enum as PgEnum,
    ForeignKey, Text
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone
from enum import Enum
import uuid

Base = declarative_base()

# Enums
class TaskType(str, Enum):
    parseDocumentPDF = "parseDocumentPDF"
    documentParsing = "documentParsing"

class TaskStatus(str, Enum):
    pending = "pending"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"

# Models
class Task(Base):
    __tablename__ = "task"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    type = Column(PgEnum(TaskType), nullable=False)
    status = Column(PgEnum(TaskStatus), default=TaskStatus.pending, nullable=False)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    createdAt = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updatedAt = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    inference_logs = relationship("InferenceLog", back_populates="task", cascade="all, delete-orphan")

class InferenceLog(Base):
    __tablename__ = "inferencelog"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    imageUrls = Column(Text, nullable=True)
    rawOutput = Column(Text, nullable=False)
    objectKeys = Column(ARRAY(String), nullable=True)
    objectUrls = Column(ARRAY(String), nullable=True)
    error = Column(Text, nullable=True)  # Thêm field này
    num_input_token = Column(Integer, nullable=False)
    num_output_token = Column(Integer, nullable=False)
    page_order = Column(Integer, nullable=True)
    createdAt = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    requestId = Column(String, ForeignKey("task.id"), nullable=True)
    task = relationship("Task", back_populates="inference_logs")
