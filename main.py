import os
import asyncio
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from bson import ObjectId
except Exception:  # fallback guard
    ObjectId = None  # type: ignore

from database import db, create_document, get_documents  # noqa: F401 (get_documents kept for future use)
from schemas import Server, Incident, UserAction, AutomationRule, MetricSnapshot

app = FastAPI(title="Server Admin Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Utilities ---------------------------------------

def to_str_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(doc)
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    # Convert datetimes to isoformat strings for JSON
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def ensure_collections():
    if db is None:
        return
    db["incident"].create_index("status")
    db["incident"].create_index("triggered_at")
    db["useraction"].create_index("incident_id")
    db["metricsnapshot"].create_index([("server", 1), ("captured_at", -1)])


def require_db():
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured. Set DATABASE_URL and DATABASE_NAME.")


# --------------------------- Seed data ---------------------------------------
DEFAULT_SERVERS: List[Server] = [
    Server(name="WebServer1", role="web", cpu=10, memory=25, disk=35, services=["nginx", "app"], status="healthy", uptime_minutes=1200),
    Server(name="DBServer1", role="db", cpu=15, memory=40, disk=50, services=["postgres"], status="healthy", uptime_minutes=2400),
    Server(name="Cache01", role="cache", cpu=5, memory=30, disk=20, services=["redis"], status="healthy", uptime_minutes=6000),
    Server(name="BackupSrv", role="backup", cpu=2, memory=15, disk=60, services=["rsnapshot"], status="healthy", uptime_minutes=3000),
]


async def seed_servers():
    if db is None:
        return
    existing = {s["name"] for s in db["server"].find({}, {"name": 1})}
    for s in DEFAULT_SERVERS:
        if s.name not in existing:
            create_document("server", s)


# --------------------------- Simulation Engine -------------------------------
_engine_task: Optional[asyncio.Task] = None
_breach_since: Dict[str, datetime] = {}  # key: f"{server}:{metric}" -> datetime


def rand_clamp(value: float, step: float, lo: float = 0.0, hi: float = 100.0) -> float:
    value += random.uniform(-step, step)
    return max(lo, min(hi, value))


def trigger_incident(server_doc: Dict[str, Any]):
    if db is None:
        return False
    choices = [
        ("high_cpu", 0.08, f"CPU spike on {server_doc['name']}"),
        ("high_memory", 0.06, f"Memory pressure on {server_doc['name']}"),
        ("disk_full", 0.03, f"Disk nearing capacity on {server_doc['name']}"),
        ("service_crash", 0.05, f"A service crashed on {server_doc['name']}")
    ]
    if server_doc.get("role") == "db":
        choices.append(("security_alert", 0.02, f"Unauthorized access attempt on {server_doc['name']}"))

    r = random.random()
    cumulative = 0.0
    for t, p, msg in choices:
        cumulative += p
        if r < cumulative:
            inc = Incident(
                server=server_doc["name"],
                type=t,  # type: ignore
                severity="high" if t in ("disk_full", "security_alert") else "medium",
                message=msg,
            )
            create_document("incident", inc)
            db["server"].update_one({"name": server_doc["name"]}, {"$set": {"status": "warning" if t != "security_alert" else "critical"}})
            return True
    return False


def apply_automation_if_needed(server_doc: Dict[str, Any]):
    if db is None:
        return
    rules = list(db["automationrule"].find({"enabled": True}))
    for rule in rules:
        if rule.get("server") != "any" and rule.get("server") != server_doc.get("role"):
            continue
        metric = rule.get("if_metric")
        operator = rule.get("operator")
        threshold = rule.get("threshold")
        value = float(server_doc.get(metric, 0.0))
        cond = (
            (operator == ">" and value > threshold) or
            (operator == ">=" and value >= threshold) or
            (operator == "<" and value < threshold) or
            (operator == "<=" and value <= threshold)
        )
        key = f"{server_doc['name']}:{metric}"
        now = datetime.now(timezone.utc)
        since = _breach_since.get(key)
        if cond:
            if not since:
                _breach_since[key] = now
                continue
            duration = (now - since).total_seconds()
            if duration >= int(rule.get("duration_sec", 60)):
                action = UserAction(
                    incident_id="auto",
                    action=rule.get("then_action"),  # type: ignore
                    performed_by="automation",
                    notes=f"Rule {rule.get('name')} triggered on {server_doc['name']} ({metric}={value:.1f}%)",
                )
                create_document("useraction", action)
                if metric in ("cpu", "memory"):
                    new_val = max(0, value - 20)
                    db["server"].update_one({"name": server_doc["name"]}, {"$set": {metric: new_val, "status": "healthy"}})
                if metric == "disk":
                    new_val = max(0, value - 15)
                    db["server"].update_one({"name": server_doc["name"]}, {"$set": {metric: new_val, "status": "healthy"}})
                type_map = {"cpu": "high_cpu", "memory": "high_memory", "disk": "disk_full"}
                itype = type_map.get(metric)
                if itype:
                    db["incident"].update_many(
                        {"server": server_doc["name"], "type": itype, "status": "active"},
                        {"$set": {"status": "resolved", "resolved_at": now}}
                    )
        else:
            if since:
                _breach_since.pop(key, None)


async def engine_loop():
    if db is None:
        return  # do not run without database
    await seed_servers()
    ensure_collections()
    while True:
        servers = list(db["server"].find({}))
        for s in servers:
            cpu = rand_clamp(float(s.get("cpu", 0)), 6)
            memory = rand_clamp(float(s.get("memory", 0)), 5)
            disk = rand_clamp(float(s.get("disk", 0)), 2)
            uptime = int(s.get("uptime_minutes", 0)) + 1
            status = "healthy"
            if max(cpu, memory, disk) > 85:
                status = "critical"
            elif max(cpu, memory, disk) > 65:
                status = "warning"
            db["server"].update_one({"_id": s["_id"]}, {"$set": {"cpu": cpu, "memory": memory, "disk": disk, "uptime_minutes": uptime, "status": status}})
            if random.random() < 0.2:
                snap = MetricSnapshot(server=s["name"], cpu=cpu, memory=memory, disk=disk)
                create_document("metricsnapshot", snap)
            if random.random() < 0.12:
                trigger_incident(s)
            apply_automation_if_needed({**s, "cpu": cpu, "memory": memory, "disk": disk})
        await asyncio.sleep(3)


@app.on_event("startup")
async def startup():
    global _engine_task
    try:
        if db is not None:
            _engine_task = asyncio.create_task(engine_loop())
        else:
            # Start without engine; endpoints that need DB will return 503
            _engine_task = None
    except Exception:
        _engine_task = None


# --------------------------- API Models --------------------------------------
class ActionRequest(BaseModel):
    action: UserAction


class AutomationRuleIn(BaseModel):
    rule: AutomationRule


# --------------------------- API Routes --------------------------------------
@app.get("/")
def root():
    return {"message": "Server Admin Simulation API running"}


@app.get("/api/servers")
def get_servers():
    require_db()
    items = [to_str_id(s) for s in db["server"].find({}).sort("name", 1)]
    return {"items": items}


@app.post("/api/servers")
def create_server(server: Server):
    require_db()
    if db["server"].find_one({"name": server.name}):
        raise HTTPException(status_code=400, detail="Server with this name already exists")
    create_document("server", server)
    return {"ok": True}


@app.get("/api/incidents")
def list_incidents(status: Optional[str] = None, limit: int = 50):
    require_db()
    filt: Dict[str, Any] = {}
    if status:
        filt["status"] = status
    items = [to_str_id(i) for i in db["incident"].find(filt).sort("triggered_at", -1).limit(limit)]
    return {"items": items}


@app.post("/api/incidents/{incident_id}/action")
def perform_action(incident_id: str, req: ActionRequest):
    require_db()
    if not ObjectId or not ObjectId.is_valid(incident_id):
        raise HTTPException(status_code=400, detail="Invalid incident id")
    inc = db["incident"].find_one({"_id": ObjectId(incident_id)})
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    create_document("useraction", req.action)
    now = datetime.now(timezone.utc)
    server_name = inc.get("server")
    if req.action.action in ("restart_service", "resolve_manual", "run_backup", "allocate_memory", "scale_out"):
        db["incident"].update_one({"_id": inc["_id"]}, {"$set": {"status": "resolved", "resolved_at": now}})
        db["server"].update_one({"name": server_name}, {"$set": {"status": "healthy"}})
    return {"ok": True}


@app.get("/api/automation-rules")
def get_rules():
    require_db()
    items = [to_str_id(r) for r in db["automationrule"].find({}).sort("name", 1)]
    return {"items": items}


@app.post("/api/automation-rules")
def create_rule(body: AutomationRuleIn):
    require_db()
    create_document("automationrule", body.rule)
    return {"ok": True}


@app.patch("/api/automation-rules/{rule_id}")
def update_rule(rule_id: str, enabled: Optional[bool] = None):
    require_db()
    if not ObjectId or not ObjectId.is_valid(rule_id):
        raise HTTPException(status_code=400, detail="Invalid rule id")
    updates: Dict[str, Any] = {}
    if enabled is not None:
        updates["enabled"] = enabled
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    db["automationrule"].update_one({"_id": ObjectId(rule_id)}, {"$set": updates})
    return {"ok": True}


@app.get("/api/analytics")
def analytics():
    require_db()
    total = db["incident"].count_documents({})
    resolved = list(db["incident"].find({"status": "resolved"}))
    active = db["incident"].count_documents({"status": "active"})
    ack = db["incident"].count_documents({"status": "acknowledged"})
    avg_resolve_sec = None
    score = 0
    if resolved:
        deltas = []
        for r in resolved:
            t1 = r.get("triggered_at")
            t2 = r.get("resolved_at")
            if isinstance(t1, datetime) and isinstance(t2, datetime):
                deltas.append((t2 - t1).total_seconds())
        if deltas:
            avg_resolve_sec = sum(deltas) / len(deltas)
            score = max(0, int(1000 - min(900, avg_resolve_sec)))
    auto_actions = db["useraction"].count_documents({"performed_by": "automation"})
    score += auto_actions * 10
    return {
        "total_incidents": total,
        "active": active,
        "acknowledged": ack,
        "resolved": len(resolved),
        "avg_resolution_seconds": avg_resolve_sec,
        "score": score,
    }


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
