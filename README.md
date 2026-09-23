# ⚡ Synapse - Fast System 1 Decision & Automation Engine

**Synapse** là hệ thống ra quyết định và tự động hóa tốc độ cao (sub-20ms) được tối ưu hóa cho Tiếng Việt, xây dựng trên nền tảng **Laya** (`convaiinnovations/laya`) kết hợp với **Rule Engine** và **Action Dispatcher (CLI & REST API)**.

---

## 🌟 Tính Năng Nổi Bật

1. **Siêu tốc & Hiệu năng cao (System 1 Inference):**
   - Sử dụng kiến trúc non-autoregressive encoder (`ModernBERT` / `mmBERT-base`) và reinforcement learning calibrated decision head (RLCD).
   - Tốc độ suy luận **dưới 20ms**, tính toán trực tiếp độ tin cậy chuẩn xác (calibrated confidence) không cần LLM cồng kềnh.

2. **Hỗ trợ Tiếng Việt & Đa tác vụ:**
   - Phân loại ý định khách hàng (`choice`), câu hỏi nhị phân (`noul` - true/false), chấm điểm mức độ nghiêm trọng (`score`).
   - Sẵn sàng script Fine-tune (`train.py`) và tự động đóng gói model xuất xưởng.

3. **Rule Engine & Tự Động Hóa Hành Động (Action Dispatcher):**
   - **Tự động thực thi:** Khớp kết quả dự đoán với quy tắc để gọi lệnh CLI hoặc gọi Webhook/REST API ngay tức thì.
   - **Xác thực nâng cao & Header động:** Hỗ trợ Bearer Token, API Key, Basic Auth và sinh header động tại runtime (HMAC SHA-256 signature, timestamp, request id).
   - **Quy trình Phê duyệt (Human-in-the-loop Approval):** Các hành động nhạy cảm hoặc mức độ nguy hiểm cao được tự động chuyển vào hàng đợi chờ người dùng phê duyệt trước khi thực thi.

---

## 📁 Cấu Trúc Thư Mục

```bash
synapse/
├── server.py           # FastAPI Server: Quản lý Rule Engine, Hàng đợi duyệt & Suy luận Laya
├── executor.py         # Module thực thi lệnh Shell (CLI) và gọi REST API (Auth & Dynamic Headers)
├── demo_executor.py    # Demo mẫu chạy thử CLI và gọi API ngoài
├── infer.py            # Script chạy suy luận trực tiếp từ terminal
├── train.py            # Script fine-tuning Laya trên tập dữ liệu tiếng Việt & đóng gói model mới
├── rules.json          # File cấu hình các quy tắc tự động hóa và phê duyệt
├── data_sample.json    # Tập dữ liệu mẫu tiếng Việt dùng để huấn luyện
└── my_laya_vi/         # Model Laya Tiếng Việt đã được huấn luyện và đóng gói
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Cài đặt môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install laya torch transformers safetensors fastapi uvicorn httpx
```

### 2. Chạy suy luận (Inference)

```bash
# Chạy với model đã huấn luyện
python infer.py --model ./my_laya_vi

# Hoặc chạy trực tiếp từ checkpoint gốc trên HuggingFace
python infer.py --model convaiinnovations/laya --subfolder multilingual
```

### 3. Huấn luyện thêm & Đóng gói Model mới

```bash
python train.py --data data_sample.json --output_dir ./my_laya_vi --epochs 5 --lr 2e-5
```

### 4. Khởi chạy API Server

```bash
python server.py
# Server mặc định chạy tại: http://localhost:8000
# Xem Swagger docs tại: http://localhost:8000/docs
```

---

## 🔌 API Endpoints Chính

| Phương thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/predict` | Chạy suy luận System 1 và tự động so khớp kích hoạt quy tắc |
| `GET` | `/rules` | Lấy danh sách tất cả các rule đã khai báo |
| `POST` | `/rules` | Thêm rule mới (CLI hoặc API action) |
| `PUT` | `/rules/{id}` | Chỉnh sửa nội dung quy tắc |
| `DELETE` | `/rules/{id}` | Xóa quy tắc |
| `GET` | `/approvals` | Lấy danh sách tác vụ đang chờ con người duyệt (`require_approval=true`) |
| `POST` | `/approvals/{id}/approve` | Phê duyệt và thực thi ngay lệnh đang chờ |
| `POST` | `/approvals/{id}/reject` | Từ chối thực thi lệnh |

---

## 📄 License
Apache-2.0 License.
