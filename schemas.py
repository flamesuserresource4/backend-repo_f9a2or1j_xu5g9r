"""
Database Schemas for the Server Admin Simulation

Each Pydantic model represents a collection in MongoDB.
Collection name is the lowercase of the class name.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime


# Users & Roles ---------------------------------------------------------------
class User(BaseModel):
    email: str = Field(..., description="Email for login")
    name: str = Field(..., description="Display name")
    role: Literal["admin", "operator"] = Field("operator", description="User role")
    is_active: bool = Field(True)


# Core domain -----------------------------------------------------------------
class Server(BaseModel):
    name: str = Field(..., description="Unique server name")
    role: Literal["web", "db", "cache", "backup", "worker", "monitor"] = Field(
        "web", description="Server function"
    )
    cpu: float = Field(0, ge=0, le=100, description="CPU usage %")
    memory: float = Field(0, ge=0, le=100, description="Memory usage %")
    disk: float = Field(0, ge=0, le=100, description="Disk usage %")
    services: List[str] = Field(default_factory=list)
    status: Literal["healthy", "warning", "critical", "down"] = Field("healthy")
    uptime_minutes: int = Field(0)


IncidentType = Literal[
    "service_crash",
    "high_cpu",
    "high_memory",
    "disk_full",
    "security_alert",
]

IncidentStatus = Literal["active", "resolved", "acknowledged"]


class Incident(BaseModel):
    server: str = Field(..., description="Server name")
    type: IncidentType
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    status: IncidentStatus = "active"
    message: str
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


class UserAction(BaseModel):
    incident_id: str
    action: Literal[
        "restart_service",
        "allocate_memory",
        "run_backup",
        "investigate_logs",
        "scale_out",
        "resolve_manual",
    ]
    performed_by: str = Field(..., description="user email or name")
    notes: Optional[str] = None
    performed_at: datetime = Field(default_factory=datetime.utcnow)
    result: Literal["success", "failure"] = "success"


# Automation ------------------------------------------------------------------
ConditionMetric = Literal["cpu", "memory", "disk"]
ActionType = Literal[
    "restart_service",
    "scale_out",
    "allocate_memory",
    "run_backup",
]


class AutomationRule(BaseModel):
    name: str
    description: Optional[str] = None
    server: Literal["any", "web", "db", "cache", "backup", "worker", "monitor"] = "any"
    if_metric: ConditionMetric = "cpu"
    operator: Literal[">", ">=", "<", "<="] = ">="
    threshold: float = Field(..., ge=0, le=100)
    duration_sec: int = Field(60, ge=5, le=3600)
    then_action: ActionType = "restart_service"
    enabled: bool = True


# Telemetry snapshot (optional collection for history) ------------------------
class MetricSnapshot(BaseModel):
    server: str
    cpu: float
    memory: float
    disk: float
    captured_at: datetime = Field(default_factory=datetime.utcnow)


# Free-form Logs ---------------------------------------------------------------
class LogEntry(BaseModel):
    level: Literal["info", "warning", "error", "critical"] = "info"
    message: str
    meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
