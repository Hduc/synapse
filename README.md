# ⚡ Synapse - Fast System 1 Decision & Automation Engine

**Synapse** là hệ thống ra quyết định và tự động hóa tốc độ cao (sub-20ms) được tối ưu hóa cho Tiếng Việt, xây dựng trên nền tảng **Laya** (`convaiinnovations/laya`) kết hợp với **Rule Engine**, **Approval Queue (Human-in-the-loop)** và **Action Dispatcher (CLI & REST API)**.

---

## 🌟 Tính Năng Nổi Bật

1. **Siêu tốc & Hiệu năng cao (System 1 Inference):**
   - Sử dụng kiến trúc non-autoregressive encoder (`ModernBERT` / `mmBERT-base`) và calibrated decision head (RLCD).
   - Tốc độ suy luận **dưới 20ms**, tính toán trực tiếp độ tin cậy chuẩn xác (calibrated confidence) không cần LLM cồng kềnh.

2. **Shortlist Cho Phân Loại Lớn (High-Cardinality Choice):**
   - Sử dụng thuật toán coarse-to-fine ranking dựa trên vector embedding của mô hình (`shortlist.predict_shortlist`).
   - Lọc nhanh top-$K$ lựa chọn phù hợp nhất trong hàng chục đến hàng trăm lựa chọn trước khi tính toán xác suất chi tiết, giải quyết giới hạn token của decision head.

3. **Thư Viện Bộ Câu Hỏi Chuẩn (Question Presets):**
   - Tích hợp sẵn 5 bộ câu hỏi chuẩn hóa chuyên dụng:
     - `triage`: Phân loại yêu cầu hỗ trợ khách hàng, độ khẩn cấp, tâm trạng bức xúc, nguy cơ rời bỏ (churn risk).
     - `email`: Phân loại mức độ ưu tiên và hành động cần thiết cho email.
     - `guard`: Phát hiện Prompt Injection, Jailbreak và bảo vệ dữ liệu nhạy cảm.
     - `moderation`: Kiểm duyệt nội dung độc hại, xúc phạm hoặc vi phạm tiêu chuẩn cộng đồng.
     - `router`: Phân loại tác vụ và định tuyến câu hỏi tới mô hình/chuyên gia phù hợp.

4. **Tiện Ích Tiền Xử Lý (Utilities):**
   - `clean-email`: Tách bỏ chữ ký, chuỗi trao đổi cũ (quote history) và định dạng email thành state chuẩn hóa cho mô hình.
   - `detect-lang`: Phân tích hệ thống chữ viết (script), tỷ lệ dấu thanh tiếng Việt (diacritic rate) và nhận diện ngôn ngữ.

5. **Rule Engine & Tự Động Hóa Hành Động (Action Dispatcher):**
   - **Tự động thực thi:** Khớp kết quả dự đoán với quy tắc để chạy lệnh CLI hoặc gọi Webhook/REST API ngay tức thì.
   - **Xác thực nâng cao & Header động:** Hỗ trợ Bearer Token, API Key, Basic Auth và hàm sinh header động tại runtime (HMAC SHA-256 signature, timestamp, request id).
   - **Quy trình Phê duyệt (Approval Workflow):** Tác vụ rủi ro cao được đưa vào hàng đợi chờ con người phê duyệt (`require_approval=true`).

---

## 📁 Cấu Trúc Thư Mục

```bash
synapse/
├── server.py           # FastAPI Server: Quản lý Rule Engine, Hàng đợi duyệt, Presets, Shortlist & Inference
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

### 2. Chạy suy luận trực tiếp (Inference CLI)

```bash
# Chạy với model đã huấn luyện cục bộ
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

## 🔌 Danh Sách API Endpoints

### 🧠 Suy Luận Quyết Định (Decision Engine)
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/predict` | Chạy suy luận System 1 chuẩn và tự động kích hoạt rules |
| `POST` | `/predict/shortlist` | Suy luận Coarse-to-Fine cho tập lựa chọn lớn ($K$ ứng viên) |

### 📋 Bộ Câu Hỏi Chuẩn (Presets)
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/presets` | Lấy danh sách các bộ preset chuẩn (`triage`, `email`, `guard`, `moderation`, `router`) |
| `GET` | `/presets/{preset_name}` | Lấy chi tiết cấu trúc câu hỏi của một preset cụ thể |

### 🛠️ Tiện Ích (Utilities)
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/utils/clean-email` | Làm sạch nội dung email thô (loại bỏ chữ ký, quote history) và tạo state |
| `POST` | `/utils/detect-lang` | Phân tích script, tỷ lệ dấu thanh (diacritic rate) tiếng Việt và ngôn ngữ |

### ⚙️ Quản Lý Quy Tắc (Rule Engine)
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/rules` | Lấy danh sách tất cả các rule (hỗ trợ lọc `active_only`, `require_approval_only`) |
| `GET` | `/rules/{id}` | Xem chi tiết quy tắc theo ID |
| `POST` | `/rules` | Thêm quy tắc mới (hành động CLI hoặc API) |
| `PUT` | `/rules/{id}` | Cập nhật cấu hình quy tắc |
| `DELETE` | `/rules/{id}` | Xóa quy tắc |
| `PATCH` | `/rules/{id}/toggle` | Bật/tắt trạng thái hoạt động hoặc chế độ duyệt |

### 🛡️ Hàng Đợi Phê Duyệt (Approval Workflow)
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/approvals` | Danh sách tác vụ đang chờ con người phê duyệt |
| `POST` | `/approvals/{id}/approve` | Phê duyệt và thực thi ngay lệnh đang chờ |
| `POST` | `/approvals/{id}/reject` | Từ chối thực thi lệnh đang chờ |

---

## 📄 License
Apache-2.0 License.
