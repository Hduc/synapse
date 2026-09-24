# ⚡ Synapse - Fast System 1 Decision & Automation Engine (v2.0.0)

**Synapse** là hệ thống ra quyết định và tự động hóa tốc độ cao (sub-20ms) được tối ưu hóa cho Tiếng Việt, xây dựng trên nền tảng **Laya** (`convaiinnovations/laya`) kết hợp với **Rule Engine**, **Approval Queue (Human-in-the-loop)**, **Dynamic UI Resolver cho Next.js**, **Task Management** và **User Behaviors Tracking**.

---

## 🌟 Tính Năng Nổi Bật (Architecture Highlights)

1. **Siêu tốc & Hiệu năng cao (System 1 Inference):**
   - Sử dụng kiến trúc non-autoregressive encoder (`mmBERT-base` / `ModernBERT`) và calibrated decision head (RLCD).
   - Tốc độ suy luận **dưới 20ms**, tính toán trực tiếp độ tin cậy chuẩn xác (calibrated confidence).

2. **Dynamic UI Resolver cho Next.js (`POST /resolve-ui`):**
   - Giải quyết bài toán: Người dùng gõ tự nhiên (vd: *"tạo form đăng ký sự kiện trung thu"*, *"xem báo cáo ai làm việc hiệu quả"*).
   - Hệ thống tự động so khớp ý định, chọn **React Component** phù hợp, trích xuất tham số/props và liên kết câu truy vấn Database tương ứng.
   - Trả về cấu trúc **JSON render tree chuẩn 100%** giúp Next.js render ngay lập tức mà không sợ crash runtime do thiếu props.

3. **Component Catalog & Props Schema API:**
   - Khai báo động danh mục Component, định nghĩa kiểu dữ liệu Props (`props_schema`), props mẫu (`sample_props`) và API kết nối (`data_binding`).

4. **Data Sources & Query Templates API:**
   - Quản lý các nguồn dữ liệu Database / REST API nội bộ.
   - Thực thi các câu truy vấn mẫu (Query Templates) theo tham số an toàn (SQL/Parametric), chống injection.

5. **Mở rộng Đa Module (Extensible Modules):**
   - **Task Module:** Quản lý danh sách công việc, tiến độ (Todo, In Progress, Review, Done), độ ưu tiên và thời hạn.
   - **User Behaviors Tracking:** Tự động ghi nhận log hành vi, câu hỏi, tương tác của người dùng để phân tích và tối ưu UX.

6. **Shortlist Cho Phân Loại Lớn (High-Cardinality Choice):**
   - Lọc nhanh top-$K$ lựa chọn phù hợp nhất bằng vector embedding trước khi tính toán xác suất chi tiết.

7. **Rule Engine & Approval Workflow:**
   - Tự động thực thi lệnh CLI / gọi REST API khi khớp điều kiện hoặc đưa vào hàng đợi chờ con người phê duyệt (`require_approval=true`).

---

## 📁 Cấu Trúc Dự Án

```bash
synapse/
├── server.py                 # FastAPI Server v2.0 (Toàn bộ REST API & Engine)
├── executor.py               # Action Dispatcher (CLI & REST API với Auth / Dynamic Headers)
├── demo_executor.py          # Demo mẫu kiểm tra thực thi CLI/API
├── infer.py                  # Script chạy suy luận trực tiếp từ terminal
├── train.py                  # Script fine-tuning Laya tiếng Việt & đóng gói model
├── rules.json                # Danh sách quy tắc tự động hóa & phê duyệt
├── components.json           # Catalog danh mục React Component, Props & Binding
├── datasources.json          # Danh mục Data Sources & Query Templates
├── tasks.json                # Dữ liệu module Quản lý Task
├── behaviors.json            # Nhật ký theo dõi hành vi người dùng
├── test_requests.http        # Toàn bộ 10 nhóm request test bằng REST Client / VS Code
├── synapse_api_collection.json # File collection import vào Thunder Client / Postman
└── my_laya_vi/               # Model Laya Tiếng Việt đã đóng gói
```

---

## 🚀 Hướng Dẫn Khởi Chạy

### 1. Cài đặt môi trường

```bash
python -m venv .venv
source .venv/bin/activate
pip install laya torch transformers safetensors fastapi uvicorn httpx pydantic
```

### 2. Khởi chạy Server

```bash
python server.py
# Server chạy mặc định tại: http://localhost:8000
# Swagger UI Docs: http://localhost:8000/docs
```

---

## 🔌 Tổng Hợp API Endpoints (v2.0)

### 🎨 1. Dynamic UI Resolver (Render Next.js)
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/resolve-ui` | Phân tích câu hỏi tự nhiên ra JSON render tree chuẩn cho Next.js |

**Ví dụ Request:**
```json
{
  "prompt": "tạo form đăng ký sự kiện trung thu",
  "user_id": "user_duc_01",
  "auto_execute_data": true
}
```
**Trả về cho Next.js:**
```json
{
  "status": "success",
  "confidence": 0.95,
  "next_action": "render",
  "render_tree": {
    "component": "EventRegistrationForm",
    "props": {
      "title": "Đăng Ký Tham Gia Lễ Hội Trăng Rằm - Trung Thu 2026",
      "event_name": "Đêm Hội Trăng Rằm - Trung Thu 2026",
      "event_date": "2026-09-25",
      "location": "Hội trường lớn & Sân khấu ngoài trời",
      "submit_button_text": "Gửi đăng ký ngay"
    },
    "data_binding": {
      "submit_api": "/api/events/register",
      "method": "POST"
    }
  }
}
```

### 🧩 2. Component Catalog (Quản lý Component & Props)
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/components` | Lấy danh mục components (lọc theo `module`) |
| `GET` | `/components/{id}` | Lấy chi tiết component, props_schema và data_binding |
| `POST` | `/components` | Khai báo thêm Component React mới |
| `PUT` | `/components/{id}` | Cập nhật cấu hình Component / Props |
| `DELETE` | `/components/{id}` | Xóa Component khỏi catalog |

### 🗄️ 3. Data Sources & Query Templates
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/datasources` | Lấy danh sách nguồn dữ liệu đã cấu hình |
| `GET` | `/datasources/{id}` | Xem chi tiết nguồn dữ liệu |
| `POST` | `/datasources` | Đăng ký thêm nguồn dữ liệu Database / REST API |
| `POST` | `/datasources/{id}/fetch` | Thực thi query template và lấy dữ liệu trả về |

### 📋 4. Task Module (Quản lý Công việc)
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/tasks` | Lấy danh sách tasks (hỗ trợ lọc `status`, `priority`) |
| `POST` | `/tasks` | Tạo task mới |
| `PUT` | `/tasks/{id}` | Cập nhật thông tin / trạng thái task |
| `DELETE` | `/tasks/{id}` | Xóa task |

### 📊 5. User Behaviors Tracking (Theo dõi Hành vi)
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/behaviors/log` | Ghi nhận sự kiện hành vi / tương tác của người dùng |
| `GET` | `/behaviors` | Lấy nhật ký hành vi người dùng (hỗ trợ lọc `user_id`, `module`) |

### 🧠 6. Decision & Shortlist Engine
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `POST` | `/predict` | Chạy suy luận System 1 chuẩn và tự động kích hoạt rules |
| `POST` | `/predict/shortlist` | Suy luận Coarse-to-Fine cho tập lựa chọn lớn ($K$ ứng viên) |

### ⚙️ 7. Rule Engine & Approval Workflow
| Method | Endpoint | Mô tả |
| :--- | :--- | :--- |
| `GET` | `/rules` | Lấy danh sách rule |
| `POST` | `/rules` | Thêm rule mới |
| `PATCH` | `/rules/{id}/toggle` | Bật/tắt rule hoặc chế độ duyệt |
| `GET` | `/approvals` | Danh sách tác vụ đang chờ duyệt |
| `POST` | `/approvals/{id}/approve` | Phê duyệt và thực thi lệnh ngay lập tức |
| `POST` | `/approvals/{id}/reject` | Từ chối thực thi lệnh |

---

## 💻 Cách Next.js Render Động từ JSON của Synapse

Trong ứng dụng Next.js, tạo Component Dynamic Renderer đơn giản:

```tsx
// components/DynamicComponentRenderer.tsx
import React from "react";
import EventRegistrationForm from "./EventRegistrationForm";
import EmployeeEfficiencyReportTable from "./EmployeeEfficiencyReportTable";
import TaskKanbanBoard from "./TaskKanbanBoard";

const COMPONENT_REGISTRY: Record<string, React.ComponentType<any>> = {
  EventRegistrationForm,
  EmployeeEfficiencyReportTable,
  TaskKanbanBoard,
};

export default function DynamicComponentRenderer({ renderTree }: { renderTree: any }) {
  if (!renderTree) return null;
  const ComponentToRender = COMPONENT_REGISTRY[renderTree.component];
  if (!ComponentToRender) {
    return <div className="p-4 border text-red-500">Component {renderTree.component} chưa đăng ký ở Frontend!</div>;
  }
  return <ComponentToRender {...renderTree.props} data={renderTree.data} />;
}
```

---

## 📄 License
Apache-2.0 License.
