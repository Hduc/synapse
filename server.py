"""FastAPI Server cho Laya Vietnamese Decision Model & Rule Engine.

Tích hợp:
1. System 1 Inference (< 20ms) & Calibrated Probabilities.
2. Rule Engine & Action Dispatcher (CLI & REST API với Auth / Dynamic Headers).
3. Hàng đợi phê duyệt (Human-in-the-loop Approval Workflow).
4. Shortlist High-Cardinality Choice (Coarse-to-Fine embedding ranking).
5. Chuẩn hóa & Cung cấp Presets câu hỏi chuẩn (triage, email, guard, moderation, router).
6. Utilities: Clean Email & Language/Diacritic Analyzer.
7. Component & Props Catalog API (Khai báo component động cho Next.js).
8. Data Sources & Query Templates API (Truy xuất database/API).
9. Dynamic UI Resolver (/resolve-ui) phân tích câu hỏi tự nhiên ra JSON render Next.js.
10. Hệ thống quản lý Tasks & Theo dõi hành vi người dùng (User Behaviors Tracking).
"""

import os
import json
import uuid
import time
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from laya import Agent, presets, shortlist, email, lang
from executor import dispatch_command

app = FastAPI(
    title="Synapse - System 1 Decision & Automation Engine",
    description="Fast non-autoregressive decision model kết hợp Rule Engine, Approval Queue, Next.js Component Resolver & Modules",
    version="2.0.0"
)

# Đường dẫn file và cấu hình
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_FILE = os.path.join(WORKSPACE_DIR, "rules.json")
COMPONENTS_FILE = os.path.join(WORKSPACE_DIR, "components.json")
DATASOURCES_FILE = os.path.join(WORKSPACE_DIR, "datasources.json")
TASKS_FILE = os.path.join(WORKSPACE_DIR, "tasks.json")
BEHAVIORS_FILE = os.path.join(WORKSPACE_DIR, "behaviors.json")

MODEL_PATH = os.environ.get("LAYA_MODEL_PATH", os.path.join(WORKSPACE_DIR, "my_laya_vi"))
SUBFOLDER = os.environ.get("LAYA_SUBFOLDER", None)
if SUBFOLDER and SUBFOLDER.lower() in ("none", "", "null"):
    SUBFOLDER = None

# Nạp model & Khởi tạo embed_fn khi khởi động
print(f"[*] Đang nạp Laya Agent từ: {MODEL_PATH}...")
try:
    agent = Agent(MODEL_PATH, subfolder=SUBFOLDER)
    embed_fn = shortlist.embed_fn_from_agent(agent)
    print(f"[+] Laya Agent đã sẵn sàng trên thiết bị: {agent.device}")
except Exception as e:
    print(f"[-] Lỗi nạp model: {e}")
    agent = None
    embed_fn = None

# In-memory store cho hàng đợi chờ duyệt (Pending Approvals)
PENDING_APPROVALS: Dict[str, Dict[str, Any]] = {}

# Map presets có sẵn của Laya
PRESET_MAP = {
    "triage": (presets.triage_questions, "Customer Support Triage: Phân loại ý định, độ khẩn cấp, cảm xúc khách hàng"),
    "email": (presets.email_questions, "Email Decision Workflow: Phân loại mức độ ưu tiên và hành động cho email"),
    "guard": (presets.guard_questions, "Safety & Guardrails: Phát hiện Prompt Injection, Jailbreak, rò rỉ dữ liệu"),
    "moderation": (presets.moderation_questions, "Content Moderation: Kiểm duyệt nội dung độc hại, xúc phạm, spam"),
    "router": (presets.router_questions, "Model & Intent Router: Định tuyến câu hỏi và phân loại tác vụ"),
}


# --- File Helpers ---
def read_json_file(file_path: str, default: Any = None) -> Any:
    if not os.path.exists(file_path):
        return [] if default is None else default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[-] Lỗi đọc {file_path}: {e}")
        return [] if default is None else default


def write_json_file(file_path: str, data: Any) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[-] Lỗi ghi {file_path}: {e}")


def load_rules_from_disk() -> List[Dict[str, Any]]:
    return read_json_file(RULES_FILE, [])


def save_rules_to_disk(rules: List[Dict[str, Any]]) -> None:
    write_json_file(RULES_FILE, rules)


def process_rule_triggers(state: Any, answers: Dict[str, Any]) -> List[Dict[str, Any]]:
    """So khớp kết quả dự đoán với các rule và thực thi hoặc đưa vào hàng đợi duyệt."""
    rules = [r for r in load_rules_from_disk() if r.get("active", True)]
    triggered_actions = []

    for rule in rules:
        trig = rule.get("trigger", {})
        q_key = trig.get("question_key")
        if q_key not in answers:
            continue

        ans = answers[q_key]
        confidence = ans.get("confidence", 0.0)
        min_conf = trig.get("min_confidence", 0.6)

        matched = False
        if "choice" in ans and "match_choice" in trig:
            if ans["choice"] == trig["match_choice"] and confidence >= min_conf:
                matched = True
        elif "noul" in ans and "match_noul" in trig:
            is_true = ans["noul"] >= 0.5
            if is_true == trig["match_noul"] and confidence >= min_conf:
                matched = True

        if matched:
            rule_id = rule["id"]
            req_approval = rule.get("require_approval", False)
            context_data = {
                "state": state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
                "choice": ans.get("choice", ""),
                "confidence": confidence,
                "rule_id": rule_id
            }

            if req_approval:
                appr_id = f"appr_{uuid.uuid4().hex[:8]}"
                ticket = {
                    "approval_id": appr_id,
                    "rule_id": rule_id,
                    "rule_name": rule.get("name"),
                    "description": rule.get("description"),
                    "action": rule["action"],
                    "context": context_data,
                    "created_at": time.time(),
                    "matched_reason": f"Khớp câu hỏi '{q_key}'={ans.get('choice')} với độ tin cậy {confidence:.2%}"
                }
                PENDING_APPROVALS[appr_id] = ticket

                triggered_actions.append({
                    "rule_id": rule_id,
                    "rule_name": rule.get("name"),
                    "status": "pending_approval",
                    "approval_id": appr_id,
                    "notice": f"Hành động này cần được phê duyệt tại POST /approvals/{appr_id}/approve",
                    "action_spec": rule["action"]
                })
            else:
                outcome = dispatch_command(rule["action"], context=context_data)
                triggered_actions.append({
                    "rule_id": rule_id,
                    "rule_name": rule.get("name"),
                    "status": "auto_executed",
                    "outcome": outcome
                })

    return triggered_actions


# --- Pydantic Schemas ---
class DecisionRequest(BaseModel):
    state: Union[str, dict, list]
    questions: Dict[str, Dict[str, Any]]
    auto_trigger: bool = True


class ShortlistDecisionRequest(BaseModel):
    state: Union[str, dict, list]
    questions: Dict[str, Dict[str, Any]]
    k: int = Field(5, description="Số lượng ứng viên tối đa giữ lại cho mỗi câu hỏi choice sau khi rank embedding")
    auto_trigger: bool = True


class EmailCleanRequest(BaseModel):
    raw_email: str = Field(..., description="Nội dung email thô bao gồm cả headers/chữ ký/quote")
    subject: Optional[str] = Field(None, description="Tiêu đề email (tùy chọn)")


class LanguageDetectRequest(BaseModel):
    text: str = Field(..., description="Đoạn văn bản cần phân tích cấu trúc ngôn ngữ/script")


class RuleTrigger(BaseModel):
    question_key: str = Field(..., description="Key câu hỏi trong predict (vd: intent)")
    match_choice: Optional[str] = Field(None, description="Giá trị choice cần so khớp")
    min_confidence: float = Field(0.6, description="Độ tin cậy tối thiểu (0.0 - 1.0)")
    match_noul: Optional[bool] = Field(None, description="Giá trị true/false cho loại câu hỏi noul")


class RuleSchema(BaseModel):
    id: Optional[str] = Field(None, description="Mã định danh rule (tự sinh nếu để trống)")
    name: str = Field(..., description="Tên quy tắc")
    description: Optional[str] = Field("", description="Mô tả quy tắc")
    trigger: RuleTrigger
    require_approval: bool = Field(False, description="True = Cần con người phê duyệt trước khi chạy")
    active: bool = Field(True, description="Trạng thái kích hoạt")
    action: Dict[str, Any] = Field(..., description="Cấu hình action (type: 'cli' hoặc 'api')")


class RuleUpdateSchema(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    trigger: Optional[RuleTrigger] = None
    require_approval: Optional[bool] = None
    active: Optional[bool] = None
    action: Optional[Dict[str, Any]] = None


# --- Schemas Mới Cho Component, Data Source, Task, Behavior & UI Resolver ---
class ComponentSchema(BaseModel):
    id: Optional[str] = Field(None, description="Mã định danh component (vd: comp_event_form)")
    name: str = Field(..., description="Tên React Component (vd: EventRegistrationForm)")
    module: str = Field("general", description="Nhóm nghiệp vụ: form, report, task, dashboard,...")
    title: str = Field(..., description="Tên hiển thị người dùng hiểu được")
    description: str = Field(..., description="Mô tả chi tiết năng lực của component")
    keywords: List[str] = Field(default_factory=list, description="Từ khóa nhận dạng để Laya so khớp")
    props_schema: Dict[str, Any] = Field(default_factory=dict, description="Định nghĩa các props & kiểu dữ liệu")
    data_binding: Optional[Dict[str, Any]] = Field(None, description="API hoặc datasource cần kết nối")
    sample_props: Optional[Dict[str, Any]] = Field(None, description="Props mẫu mặc định để render ngay")


class DataSourceSchema(BaseModel):
    id: Optional[str] = Field(None, description="ID nguồn dữ liệu (vd: ds_employee_kpi)")
    name: str = Field(..., description="Tên nguồn dữ liệu")
    type: str = Field("database", description="database hoặc rest_api")
    description: Optional[str] = ""
    tables: Optional[List[str]] = Field(default_factory=list)
    query_templates: Optional[Dict[str, str]] = Field(default_factory=dict)
    mock_data: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class TaskSchema(BaseModel):
    id: Optional[str] = None
    title: str = Field(..., description="Tiêu đề task")
    description: Optional[str] = ""
    assigned_to: Optional[str] = "Chưa gán"
    status: str = Field("todo", description="todo, in_progress, review, done")
    priority: str = Field("medium", description="low, medium, high, urgent")
    due_date: Optional[str] = None


class UserBehaviorLog(BaseModel):
    user_id: Optional[str] = "anonymous"
    action: str = Field(..., description="Hành động: query_prompt, view_component, submit_form, click_button")
    module: Optional[str] = "general"
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ResolveUIRequest(BaseModel):
    prompt: str = Field(..., description="Câu hỏi hoặc yêu cầu từ người dùng (vd: tạo form đăng ký sự kiện trung thu)")
    user_id: Optional[str] = Field("user_default", description="Mã người dùng thực hiện")
    module_filter: Optional[str] = Field(None, description="Lọc theo module nếu có (vd: form, report, task)")
    auto_execute_data: bool = Field(True, description="Tự động truy vấn dữ liệu mẫu nếu component cần")


# --- Endpoints Hệ thống ---
@app.get("/")
def index():
    rules = load_rules_from_disk()
    components = read_json_file(COMPONENTS_FILE, [])
    datasources = read_json_file(DATASOURCES_FILE, [])
    tasks = read_json_file(TASKS_FILE, [])
    behaviors = read_json_file(BEHAVIORS_FILE, [])
    return {
        "status": "online",
        "service": "Synapse - System 1 Decision & Automation Engine",
        "version": "2.0.0",
        "model": MODEL_PATH,
        "device": str(agent.device) if agent else "unavailable",
        "shortlist_enabled": embed_fn is not None,
        "total_rules": len(rules),
        "total_components": len(components),
        "total_datasources": len(datasources),
        "total_tasks": len(tasks),
        "total_behaviors_logged": len(behaviors),
        "pending_approvals": len(PENDING_APPROVALS),
        "available_presets": list(PRESET_MAP.keys())
    }


@app.get("/health")
def health():
    if agent is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")
    return {"status": "healthy"}


# ==========================================
# 1. COMPONENT CATALOG API (KHAI BÁO UI NEXT.JS)
# ==========================================

@app.get("/components", summary="Lấy danh mục các components đã khai báo")
def get_components(module: Optional[str] = None):
    components = read_json_file(COMPONENTS_FILE, [])
    if module:
        components = [c for c in components if c.get("module") == module]
    return {
        "count": len(components),
        "components": components
    }


@app.get("/components/{comp_id}", summary="Lấy chi tiết cấu hình 1 component")
def get_component_detail(comp_id: str):
    components = read_json_file(COMPONENTS_FILE, [])
    for c in components:
        if c["id"] == comp_id:
            return c
    raise HTTPException(status_code=404, detail=f"Không tìm thấy component '{comp_id}'")


@app.post("/components", summary="Khai báo thêm một Component mới cho Next.js")
def create_component(comp: ComponentSchema):
    components = read_json_file(COMPONENTS_FILE, [])
    comp_dict = comp.model_dump()
    if not comp_dict.get("id"):
        comp_dict["id"] = f"comp_{uuid.uuid4().hex[:8]}"

    if any(c["id"] == comp_dict["id"] for c in components):
        raise HTTPException(status_code=400, detail=f"Component ID '{comp_dict['id']}' đã tồn tại")

    components.append(comp_dict)
    write_json_file(COMPONENTS_FILE, components)
    return {
        "status": "created",
        "message": f"Đã đăng ký component '{comp_dict['name']}' thành công",
        "component": comp_dict
    }


@app.put("/components/{comp_id}", summary="Cập nhật Component, Props hoặc Data Binding")
def update_component(comp_id: str, comp: ComponentSchema):
    components = read_json_file(COMPONENTS_FILE, [])
    idx = None
    for i, c in enumerate(components):
        if c["id"] == comp_id:
            idx = i
            break

    if idx is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy component '{comp_id}'")

    comp_dict = comp.model_dump(exclude_unset=True)
    comp_dict["id"] = comp_id
    components[idx] = {**components[idx], **comp_dict}
    write_json_file(COMPONENTS_FILE, components)
    return {
        "status": "updated",
        "message": f"Đã cập nhật component '{comp_id}'",
        "component": components[idx]
    }


@app.delete("/components/{comp_id}", summary="Xóa một component khỏi catalog")
def delete_component(comp_id: str):
    components = read_json_file(COMPONENTS_FILE, [])
    filtered = [c for c in components if c["id"] != comp_id]
    if len(filtered) == len(components):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy component '{comp_id}'")
    write_json_file(COMPONENTS_FILE, filtered)
    return {"status": "deleted", "message": f"Đã xóa component '{comp_id}'"}


# ==========================================
# 2. DATA SOURCES & QUERY TEMPLATES API
# ==========================================

@app.get("/datasources", summary="Lấy danh sách các Data Sources đã cấu hình")
def get_datasources():
    datasources = read_json_file(DATASOURCES_FILE, [])
    return {
        "count": len(datasources),
        "datasources": datasources
    }


@app.get("/datasources/{ds_id}", summary="Xem chi tiết Data Source")
def get_datasource_detail(ds_id: str):
    datasources = read_json_file(DATASOURCES_FILE, [])
    for ds in datasources:
        if ds["id"] == ds_id:
            return ds
    raise HTTPException(status_code=404, detail=f"Không tìm thấy Data Source '{ds_id}'")


@app.post("/datasources", summary="Đăng ký thêm một Data Source mới")
def create_datasource(ds: DataSourceSchema):
    datasources = read_json_file(DATASOURCES_FILE, [])
    ds_dict = ds.model_dump()
    if not ds_dict.get("id"):
        ds_dict["id"] = f"ds_{uuid.uuid4().hex[:8]}"

    if any(d["id"] == ds_dict["id"] for d in datasources):
        raise HTTPException(status_code=400, detail=f"Data Source '{ds_dict['id']}' đã tồn tại")

    datasources.append(ds_dict)
    write_json_file(DATASOURCES_FILE, datasources)
    return {"status": "created", "datasource": ds_dict}


@app.post("/datasources/{ds_id}/fetch", summary="Truy xuất dữ liệu từ nguồn theo query template")
def fetch_datasource_data(ds_id: str, payload: Dict[str, Any] = None):
    datasources = read_json_file(DATASOURCES_FILE, [])
    target_ds = None
    for ds in datasources:
        if ds["id"] == ds_id:
            target_ds = ds
            break

    if not target_ds:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy Data Source '{ds_id}'")

    mock = target_ds.get("mock_data", [])
    query = payload.get("query") if payload else None
    template_key = payload.get("template_key") if payload else "top_performers"
    templates = target_ds.get("query_templates", {})
    resolved_query = query or templates.get(template_key, "SELECT * FROM default_table")

    return {
        "status": "success",
        "datasource_id": ds_id,
        "datasource_name": target_ds["name"],
        "executed_query": resolved_query,
        "rows_count": len(mock),
        "data": mock
    }


# ==========================================
# 3. TASK MODULE API
# ==========================================

@app.get("/tasks", summary="Lấy danh sách các task công việc (hỗ trợ lọc status, priority)")
def get_tasks(status: Optional[str] = None, priority: Optional[str] = None):
    tasks = read_json_file(TASKS_FILE, [])
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    if priority:
        tasks = [t for t in tasks if t.get("priority") == priority]
    return {
        "count": len(tasks),
        "tasks": tasks
    }


@app.post("/tasks", summary="Tạo task mới")
def create_task(task: TaskSchema):
    tasks = read_json_file(TASKS_FILE, [])
    t_dict = task.model_dump()
    if not t_dict.get("id"):
        t_dict["id"] = f"task_{uuid.uuid4().hex[:8]}"
    t_dict["created_at"] = time.time()
    tasks.append(t_dict)
    write_json_file(TASKS_FILE, tasks)
    return {"status": "created", "task": t_dict}


@app.put("/tasks/{task_id}", summary="Cập nhật trạng thái hoặc thông tin task")
def update_task(task_id: str, patch: Dict[str, Any]):
    tasks = read_json_file(TASKS_FILE, [])
    target = None
    for t in tasks:
        if t["id"] == task_id:
            target = t
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy task '{task_id}'")

    for k, v in patch.items():
        if k != "id":
            target[k] = v

    write_json_file(TASKS_FILE, tasks)
    return {"status": "updated", "task": target}


@app.delete("/tasks/{task_id}", summary="Xóa một task")
def delete_task(task_id: str):
    tasks = read_json_file(TASKS_FILE, [])
    filtered = [t for t in tasks if t["id"] != task_id]
    if len(filtered) == len(tasks):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy task '{task_id}'")
    write_json_file(TASKS_FILE, filtered)
    return {"status": "deleted", "message": f"Đã xóa task '{task_id}'"}


# ==========================================
# 4. USER BEHAVIORS TRACKING API
# ==========================================

@app.post("/behaviors/log", summary="Ghi nhận hành vi / tương tác của người dùng")
def log_user_behavior(log: UserBehaviorLog):
    behaviors = read_json_file(BEHAVIORS_FILE, [])
    entry = {
        "id": f"beh_{uuid.uuid4().hex[:8]}",
        "user_id": log.user_id,
        "action": log.action,
        "module": log.module,
        "details": log.details,
        "timestamp": time.time()
    }
    behaviors.append(entry)
    # Giới hạn lưu 5000 sự kiện gần nhất
    if len(behaviors) > 5000:
        behaviors = behaviors[-5000:]
    write_json_file(BEHAVIORS_FILE, behaviors)
    return {"status": "logged", "entry_id": entry["id"]}


@app.get("/behaviors", summary="Lấy lịch sử hành vi người dùng")
def get_user_behaviors(user_id: Optional[str] = None, module: Optional[str] = None, limit: int = 50):
    behaviors = read_json_file(BEHAVIORS_FILE, [])
    if user_id:
        behaviors = [b for b in behaviors if b.get("user_id") == user_id]
    if module:
        behaviors = [b for b in behaviors if b.get("module") == module]
    behaviors.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return {
        "count": min(len(behaviors), limit),
        "total": len(behaviors),
        "behaviors": behaviors[:limit]
    }


# ==========================================
# 5. DYNAMIC UI RESOLVER (LỰA CHỌN COMPONENT & PROPS CHO NEXT.JS)
# ==========================================

@app.post("/resolve-ui", summary="Phân tích yêu cầu tự nhiên và tạo JSON render chuẩn cho Next.js")
def resolve_ui(req: ResolveUIRequest):
    """Giải quyết bài toán:
    1. Tiếp nhận prompt (vd: 'tạo form đăng ký sự kiện trung thu' hoặc 'xem báo cáo ai làm việc hiệu quả').
    2. Chạy Laya System 1 so khớp với danh mục Component và Intent.
    3. Trích xuất props, parameters và câu truy vấn tương ứng.
    4. Tự động liên kết Data Source nếu component cần dữ liệu.
    5. Ghi log hành vi người dùng và trả về JSON chuẩn để Next.js render 100% không crash.
    """
    components = read_json_file(COMPONENTS_FILE, [])
    if req.module_filter:
        components = [c for c in components if c.get("module") == req.module_filter]

    if not components:
        raise HTTPException(status_code=404, detail="Không có component nào khả dụng để phân tích")

    prompt_lower = req.prompt.lower().strip()

    # Dùng Laya để phân loại ý định chính
    intent_criteria = {
        c["id"]: f"{c['title']}: {c['description']}"
        for c in components
    }

    questions = {
        "selected_component": {
            "type": "choice",
            "instructions": f"Chọn component UI phù hợp nhất để hiển thị cho yêu cầu: '{req.prompt}'",
            "criteria": intent_criteria
        },
        "is_confident": {
            "type": "noul",
            "instructions": f"Yêu cầu '{req.prompt}' có cung cấp đủ ngữ cảnh để chọn component không?"
        }
    }

    # 1. So khớp từ khóa / cụm từ nghiệp vụ (Trọng số ưu tiên cao cho intent cụ thể)
    matched_comp = None
    confidence = 0.85
    best_score = 0

    for c in components:
        kws = c.get("keywords", [])
        hits = [kw for kw in kws if kw in prompt_lower]
        score = sum(len(kw.split()) for kw in hits)
        if score > best_score:
            best_score = score
            matched_comp = c
            confidence = 0.95

    # 2. Nếu không khớp từ khóa rõ ràng, dùng Laya Zero-Shot Decision Model
    if not matched_comp and agent:
        try:
            res = agent.predict(req.prompt, questions)
            ans = res["answers"]["selected_component"]
            predicted_id = ans.get("choice")
            confidence = ans.get("confidence", 0.80)

            for c in components:
                if c["id"] == predicted_id:
                    matched_comp = c
                    break
        except Exception as e:
            print(f"[-] Laya inference warning: {e}")

    if not matched_comp:
        matched_comp = components[0]
        confidence = 0.5

    # Sinh props tương ứng dựa trên prompt
    resolved_props = dict(matched_comp.get("sample_props", {}))
    missing_slots = []

    # Xử lý form đăng ký sự kiện
    if matched_comp["name"] == "EventRegistrationForm":
        if "trung thu" in prompt_lower:
            resolved_props["title"] = "Đăng Ký Tham Gia Lễ Hội Trăng Rằm - Trung Thu 2026"
            resolved_props["event_name"] = "Đêm Hội Trăng Rằm - Trung Thu 2026"
            resolved_props["event_date"] = "2026-09-25"
            resolved_props["location"] = "Hội trường lớn & Sân khấu ngoài trời"
        elif "hội thảo" in prompt_lower:
            resolved_props["title"] = "Đăng Ký Tham Dự Hội Thảo Công Nghệ"
            resolved_props["event_name"] = "Hội Thảo Công Nghệ 2026"
        else:
            resolved_props["title"] = f"Đăng Ký Sự Kiện: {req.prompt.capitalize()}"

    # Xử lý báo cáo hiệu quả nhân viên
    elif matched_comp["name"] == "EmployeeEfficiencyReportTable":
        if "ai làm việc hiệu quả" in prompt_lower or "hiệu quả" in prompt_lower:
            resolved_props["title"] = "Bảng Xếp Hạng Hiệu Quả Làm Việc & KPI Nhân Viên"
            resolved_props["sort_by"] = "kpi_score_desc"
            resolved_props["period"] = "Tháng 09/2026"

    # Xử lý data binding
    data_payload = None
    data_binding = matched_comp.get("data_binding", {})
    if req.auto_execute_data and data_binding:
        fetch_api = data_binding.get("fetch_api")
        if fetch_api and "ds_employee_kpi" in fetch_api:
            datasources = read_json_file(DATASOURCES_FILE, [])
            for ds in datasources:
                if ds["id"] == "ds_employee_kpi":
                    data_payload = {
                        "executed_query": data_binding.get("query_template"),
                        "rows": ds.get("mock_data", [])
                    }
                    break
        elif fetch_api and "tasks" in fetch_api:
            tasks = read_json_file(TASKS_FILE, [])
            data_payload = {
                "tasks": tasks
            }

    # Ghi nhận hành vi người dùng
    log_user_behavior(UserBehaviorLog(
        user_id=req.user_id,
        action="resolve_ui",
        module=matched_comp.get("module", "general"),
        details={
            "prompt": req.prompt,
            "resolved_component": matched_comp["name"],
            "confidence": confidence
        }
    ))

    # Cấu trúc JSON chuẩn chỉnh để Next.js render
    render_tree = {
        "component": matched_comp["name"],
        "component_id": matched_comp["id"],
        "module": matched_comp.get("module"),
        "title": matched_comp.get("title"),
        "props": resolved_props,
        "props_schema": matched_comp.get("props_schema"),
        "data_binding": data_binding,
        "data": data_payload,
        "timestamp": time.time()
    }

    return {
        "status": "success",
        "prompt": req.prompt,
        "confidence": round(confidence, 4),
        "next_action": "render",
        "render_tree": render_tree,
        "missing_slots": missing_slots
    }


# ==========================================
# 6. PRESETS API
# ==========================================

@app.get("/presets", summary="Lấy danh sách các bộ câu hỏi mẫu chuẩn hóa của Laya")
def list_presets():
    return {
        "count": len(PRESET_MAP),
        "presets": [
            {
                "id": k,
                "name": k.capitalize(),
                "description": desc
            }
            for k, (fn, desc) in PRESET_MAP.items()
        ]
    }


@app.get("/presets/{preset_name}", summary="Lấy chi tiết câu hỏi của một preset")
def get_preset_detail(preset_name: str):
    key = preset_name.lower().strip()
    if key not in PRESET_MAP:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{preset_name}' không tồn tại. Các preset hỗ trợ: {list(PRESET_MAP.keys())}"
        )
    fn, desc = PRESET_MAP[key]
    return {
        "preset": key,
        "description": desc,
        "questions": fn()
    }


# ==========================================
# 7. UTILITIES API (EMAIL & LANGUAGE)
# ==========================================

@app.post("/utils/clean-email", summary="Tiền xử lý và làm sạch nội dung email thô")
def clean_email_endpoint(req: EmailCleanRequest):
    cleaned_body = email.clean_email_body(req.raw_email)
    state = email.email_state(subject=req.subject or "", body=req.raw_email, clean=True)
    return {
        "cleaned_body": cleaned_body,
        "state": state
    }


@app.post("/utils/detect-lang", summary="Phân tích script, tỷ lệ dấu thanh tiếng Việt và ngôn ngữ")
def detect_lang_endpoint(req: LanguageDetectRequest):
    analysis = lang.analyse(req.text)
    return {
        "text": req.text,
        "analysis": analysis
    }


# ==========================================
# 8. QUẢN LÝ RULE (LẤY, THÊM, SỬA, XÓA)
# ==========================================

@app.get("/rules", summary="Lấy danh sách tất cả các rule đã khai báo")
def get_rules(active_only: bool = False, require_approval_only: bool = False):
    rules = load_rules_from_disk()
    if active_only:
        rules = [r for r in rules if r.get("active", True)]
    if require_approval_only:
        rules = [r for r in rules if r.get("require_approval", False)]
    return {
        "count": len(rules),
        "rules": rules
    }


@app.get("/rules/{rule_id}", summary="Xem chi tiết một rule theo ID")
def get_rule(rule_id: str):
    rules = load_rules_from_disk()
    for r in rules:
        if r["id"] == rule_id:
            return r
    raise HTTPException(status_code=404, detail=f"Không tìm thấy rule '{rule_id}'")


@app.post("/rules", summary="Khai báo thêm một rule mới")
def create_rule(rule: RuleSchema):
    rules = load_rules_from_disk()
    rule_dict = rule.model_dump()
    if not rule_dict.get("id"):
        rule_dict["id"] = f"rule_{uuid.uuid4().hex[:8]}"

    if any(r["id"] == rule_dict["id"] for r in rules):
        raise HTTPException(status_code=400, detail=f"Rule ID '{rule_dict['id']}' đã tồn tại")

    rules.append(rule_dict)
    save_rules_to_disk(rules)
    return {
        "status": "created",
        "message": f"Đã thêm rule thành công",
        "rule": rule_dict
    }


@app.put("/rules/{rule_id}", summary="Chỉnh sửa / Cập nhật một rule đã có")
def update_rule(rule_id: str, patch: RuleUpdateSchema):
    rules = load_rules_from_disk()
    target_idx = None
    for i, r in enumerate(rules):
        if r["id"] == rule_id:
            target_idx = i
            break

    if target_idx is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy rule '{rule_id}' để sửa")

    current_rule = rules[target_idx]
    patch_data = patch.model_dump(exclude_unset=True)

    for k, v in patch_data.items():
        if k == "trigger" and v is not None:
            current_rule["trigger"] = v
        else:
            current_rule[k] = v

    rules[target_idx] = current_rule
    save_rules_to_disk(rules)
    return {
        "status": "updated",
        "message": f"Đã cập nhật rule '{rule_id}'",
        "rule": current_rule
    }


@app.delete("/rules/{rule_id}", summary="Xóa một rule")
def delete_rule(rule_id: str):
    rules = load_rules_from_disk()
    filtered = [r for r in rules if r["id"] != rule_id]
    if len(filtered) == len(rules):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy rule '{rule_id}' để xóa")

    save_rules_to_disk(filtered)
    return {
        "status": "deleted",
        "message": f"Đã xóa rule '{rule_id}'"
    }


@app.patch("/rules/{rule_id}/toggle", summary="Bật/Tắt trạng thái hoạt động hoặc yêu cầu phê duyệt")
def toggle_rule(rule_id: str, active: Optional[bool] = None, require_approval: Optional[bool] = None):
    rules = load_rules_from_disk()
    for r in rules:
        if r["id"] == rule_id:
            if active is not None:
                r["active"] = active
            if require_approval is not None:
                r["require_approval"] = require_approval
            save_rules_to_disk(rules)
            return {"status": "success", "rule": r}
    raise HTTPException(status_code=404, detail=f"Không tìm thấy rule '{rule_id}'")


# ==========================================
# 9. APPROVAL WORKFLOW (HÀNG ĐỢI PHÊ DUYỆT)
# ==========================================

@app.get("/approvals", summary="Lấy danh sách các lệnh đang chờ người dùng phê duyệt")
def get_pending_approvals():
    return {
        "count": len(PENDING_APPROVALS),
        "pending": list(PENDING_APPROVALS.values())
    }


@app.post("/approvals/{approval_id}/approve", summary="Duyệt và thực thi ngay lệnh đang chờ")
def approve_action(approval_id: str):
    if approval_id not in PENDING_APPROVALS:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ticket chờ duyệt '{approval_id}'")

    ticket = PENDING_APPROVALS.pop(approval_id)
    action_spec = ticket["action"]
    context = ticket.get("context", {})

    outcome = dispatch_command(action_spec, context=context)
    return {
        "status": "approved_and_executed",
        "approval_id": approval_id,
        "rule_id": ticket["rule_id"],
        "rule_name": ticket["rule_name"],
        "execution_outcome": outcome
    }


@app.post("/approvals/{approval_id}/reject", summary="Từ chối thực thi lệnh đang chờ")
def reject_action(approval_id: str, reason: Optional[str] = "Người dùng từ chối"):
    if approval_id not in PENDING_APPROVALS:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ticket chờ duyệt '{approval_id}'")

    ticket = PENDING_APPROVALS.pop(approval_id)
    return {
        "status": "rejected",
        "approval_id": approval_id,
        "rule_id": ticket["rule_id"],
        "reason": reason
    }


# ==========================================
# 10. PREDICT & SHORTLIST DECISION ENGINE
# ==========================================

@app.post("/predict", summary="Chạy suy luận Laya tiêu chuẩn và kích hoạt rules")
def predict(req: DecisionRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    try:
        decision_result = agent.predict(req.state, req.questions)
        triggered_actions = []

        if req.auto_trigger:
            triggered_actions = process_rule_triggers(req.state, decision_result.get("answers", {}))

        return {
            **decision_result,
            "triggered_actions": triggered_actions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/shortlist", summary="Suy luận Coarse-to-Fine cho tập lựa chọn lớn (High-Cardinality)")
def predict_shortlist_endpoint(req: ShortlistDecisionRequest):
    if agent is None or embed_fn is None:
        raise HTTPException(status_code=503, detail="Model hoặc Embedder chưa sẵn sàng")

    try:
        shortlist_result = shortlist.predict_shortlist(
            agent=agent,
            state=req.state,
            questions=req.questions,
            embed_fn=embed_fn,
            k=req.k
        )

        triggered_actions = []
        if req.auto_trigger:
            triggered_actions = process_rule_triggers(req.state, shortlist_result.get("answers", {}))

        return {
            **shortlist_result,
            "triggered_actions": triggered_actions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
