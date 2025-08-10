# services/create_task.py
from datetime import datetime, timezone
from typing import List
import uuid
from sqlalchemy.orm import Session
from app.db.models import Task, TaskStatus, TaskType
from app.db.models import InferenceLog

def create_task(task_type: TaskType, session: Session) -> Task:
    new_task = Task(
        type=task_type,
        status=TaskStatus.pending,
    )
    session.add(new_task)
    session.commit()
    session.refresh(new_task)
    return new_task

def create_log(imageUrls : str,
            objectKeys : List[str],
            objectUrls : List[str],
            requestId : str,
            num_input_token : int,
            num_output_token : int,
            rawOutput : str,
            page_order : int,
            error: str,
            session : Session):
    new_log = InferenceLog(
        imageUrls=imageUrls,
        objectKeys=objectKeys,
        objectUrls=objectUrls,
        requestId=requestId,
        num_input_token=num_input_token,
        num_output_token=num_output_token,
        rawOutput=rawOutput,
        error=error,
        page_order=page_order,
    )
    session.add(new_log)
    session.commit()
    session.refresh(new_log)
    return new_log

def get_task_by_id(session : Session, task_id: str):
    return session.query(Task).filter(Task.id == task_id).first()

def get_logs_by_task(session: Session, task_id: str) -> List[InferenceLog]:
    return session.query(InferenceLog).filter(InferenceLog.requestId == task_id).all()

def update_task_result(session : Session, task_id: str, result : str):
    task = session.query(Task).filter(Task.id == task_id).first()
    task.result = result
    task.updatedAt = datetime.now(timezone.utc)
    task.status = TaskStatus.done