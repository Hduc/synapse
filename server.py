"""FastAPI Server cho Laya Vietnamese Decision Model & Rule Engine."""

import os
import json
import uuid
import time
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from laya import Agent
from executor import dispatch_command

app = FastAPI(
    title="Laya Decision & Rule Automation API",
    description="System 1 Decision Model kết hợp Rule Engine & Approval Workflow (CLI & REST API)",
    version="1.1.0"
)

# Đường dẫn file và cấu hình
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_FILE = os.path.join(WORKSPACE_DIR, "rules.json")
MODEL_PATH = os.environ.get("LAYA_MODEL_PATH", os.path.join(WORKSPACE_DIR, "my_laya_vi"))
SUBFOLDER = os.environ.get("LAYA_SUBFOLDER", None)
if SUBFOLDER and SUBFOLDER.lower() in ("none", "", "null"):
    SUBFOLDER = None

# Nạp model khi khởi động
print(f"[*] Đang nạp Laya Agent từ: {MODEL_PATH}...")
try:
    agent = Agent(MODEL_PATH, subfolder=SUBFOLDER)
    print(f"[+] Laya Agent đã sẵn sàng trên: {agent.device}")
except Exception as e:
    print(f"[-] Lỗi nạp model: {e}")
    agent = None

# In-memory store cho hàng đợi chờ duyệt (Pending Approvals)
PENDING_APPROVALS: Dict[str, Dict[str, Any]] = {}


# --- Helper Quản lý Rules File ---
def load_rules_from_disk() -> List[Dict[str, Any]]:
    if not os.path.exists(RULES_FILE):
        return []
    try:
        with open(RULES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[-] Lỗi đọc rules.json: {e}")
        return []


def save_rules_to_disk(rules: List[Dict[str, Any]]) -> None:
    with open(RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)


# --- Schemas ---
class DecisionRequest(BaseModel):
    state: Union[str, dict, list]
    questions: Dict[str, Dict[str, Any]]
    auto_trigger: bool = True  # Tự động so khớp và kích hoạt rules


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


# --- Endpoints Hệ thống ---
@app.get("/")
def index():
    rules = load_rules_from_disk()
    return {
        "status": "online",
        "service": "Laya Vietnamese Decision & Automation Engine",
        "model": MODEL_PATH,
        "device": str(agent.device) if agent else "unavailable",
        "total_rules": len(rules),
        "pending_approvals": len(PENDING_APPROVALS)
    }


@app.get("/health")
def health():
    if agent is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")
    return {"status": "healthy"}


# ==========================================
# 1. API QUẢN LÝ RULE (LẤY, THÊM, SỬA, XÓA)
# ==========================================

@app.get("/rules", summary="Lấy danh sách tất cả các rule đã khai báo")
def get_rules(active_only: bool = False, require_approval_only: bool = False):
    """Lấy danh sách các rule.

    Có thể lọc theo `active_only=true` hoặc `require_approval_only=true`.
    """
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

    # Kiểm tra trùng ID
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

    # Cập nhật các trường
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
# 2. API PHÊ DUYỆT (APPROVAL WORKFLOW)
# ==========================================

@app.get("/approvals", summary="Lấy danh sách các lệnh đang chờ người dùng phê duyệt")
def get_pending_approvals():
    """Danh sách các lệnh do model kích hoạt nhưng bị chặn lại vì require_approval=True."""
    return {
        "count": len(PENDING_APPROVALS),
        "pending": list(PENDING_APPROVALS.values())
    }


@app.post("/approvals/{approval_id}/approve", summary="Duyệt và thực thi ngay lệnh đang chờ")
def approve_action(approval_id: str):
    """Phê duyệt một lệnh: Hệ thống sẽ lấy action spec và gọi CLI hoặc API ngay lập tức."""
    if approval_id not in PENDING_APPROVALS:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ticket chờ duyệt '{approval_id}'")

    ticket = PENDING_APPROVALS.pop(approval_id)
    action_spec = ticket["action"]
    context = ticket.get("context", {})

    # Thực thi lệnh
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
# 3. PREDICT & RULE MATCHING ENGINE
# ==========================================

@app.post("/predict", summary="Chạy suy luận Laya và tự động kích hoạt / đưa vào hàng đợi duyệt")
def predict(req: DecisionRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    try:
        # 1. Chạy suy luận tốc độ cao bằng Laya
        decision_result = agent.predict(req.state, req.questions)
        answers = decision_result.get("answers", {})

        triggered_actions = []

        # 2. So khớp rules nếu bật auto_trigger
        if req.auto_trigger:
            rules = [r for r in load_rules_from_disk() if r.get("active", True)]

            for rule in rules:
                trig = rule.get("trigger", {})
                q_key = trig.get("question_key")
                if q_key not in answers:
                    continue

                ans = answers[q_key]
                confidence = ans.get("confidence", 0.0)
                min_conf = trig.get("min_confidence", 0.6)

                matched = False
                # So khớp Choice
                if "choice" in ans and "match_choice" in trig:
                    if ans["choice"] == trig["match_choice"] and confidence >= min_conf:
                        matched = True

                # So khớp Noul (True/False)
                elif "noul" in ans and "match_noul" in trig:
                    is_true = ans["noul"] >= 0.5
                    if is_true == trig["match_noul"] and confidence >= min_conf:
                        matched = True

                if matched:
                    rule_id = rule["id"]
                    req_approval = rule.get("require_approval", False)
                    context_data = {
                        "state": req.state if isinstance(req.state, str) else json.dumps(req.state, ensure_ascii=False),
                        "choice": ans.get("choice", ""),
                        "confidence": confidence,
                        "rule_id": rule_id
                    }

                    if req_approval:
                        # Tạo ticket chờ phê duyệt
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
                        # Tự động thực thi ngay lập tức
                        outcome = dispatch_command(rule["action"], context=context_data)
                        triggered_actions.append({
                            "rule_id": rule_id,
                            "rule_name": rule.get("name"),
                            "status": "auto_executed",
                            "outcome": outcome
                        })

        return {
            **decision_result,
            "triggered_actions": triggered_actions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
