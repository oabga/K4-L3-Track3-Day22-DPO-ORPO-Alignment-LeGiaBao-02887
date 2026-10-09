# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Lê Gia Bảo
**Khoá:** K4
**Tier đã chạy:** T4 (NB0, NB2 xong; NB1/NB3/NB4 **chưa chạy**, xem ghi chú bên dưới)
**Ngày:** 2026-10-09

> **Ghi chú tình trạng nộp bài:** Tới hạn nộp, cả hai tài khoản Google Colab dùng để chạy lab đều đã hết
> quota GPU T4 miễn phí (lỗi "Cannot connect to GPU backend" sau khi chạy được một phần NB1/NB3 trong phiên
> trước, phiên bị ngắt và mất file vì Colab xoá `/content` khi hết phiên). Kaggle Notebooks (lựa chọn thay thế,
> 30 giờ T4/tuần) cũng đã dùng hết quota trong cùng ngày. Máy cá nhân không đủ điều kiện chạy NB1/NB3/NB4
> (GPU 4 GB VRAM, lab yêu cầu tối thiểu 12 GB; xem `HARDWARE-GUIDE.md`).
>
> **NB0** (viết `my_dpo_loss`, hai câu hỏi lý thuyết) và **NB2** (chia dữ liệu sở thích, đo thiên vị độ dài)
> không cần GPU nên đã chạy thật, kết quả thật ở `data/pref/stats.json` và `submission/screenshots/02b-pref-length.png`.
> **NB1 (SFT), NB3 (DPO), NB4 (chấm tự động)** cần GPU thật nên chưa có số liệu — các mục tương ứng bên dưới
> (§2–§4) để trống thay vì điền số ước lượng, vì bài chấm theo số liệu thật từ file do notebook sinh ra.
> Dự kiến chạy lại và bổ sung khi quota GPU miễn phí được cấp lại (thường trong vòng 24 giờ).

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB (dự kiến; NB1/NB3 chưa chạy được do hết quota GPU free — xem ghi chú đầu file) |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` (cấu hình tier T4, `lab22/config.py`) |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · chưa chạy (NB1 chưa thực hiện) |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out — **đã chạy thật (NB2)** |
| Chosen dài hơn rejected (NB2) | **65,9%** (chosen median 94 token, rejected median 86 token) — số thật từ `data/pref/stats.json` |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 (giá trị cấu hình mặc định tier T4; NB3 chưa chạy nên chưa có kết quả) |
| Giám khảo | rm: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B (mặc định); NB4 chưa chạy |
| Chi phí | 0 đồng (Colab + Kaggle free tier); cả hai đã hết quota GPU trước khi hoàn thành NB1/NB3/NB4 |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | _<...>_ |
| VRAM cao nhất | _<...>_ |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | _<...>_ |
| Độ chính xác reward trên held-out | _<...>_ |
| Margin trên held-out | _<...>_ |
| Chẩn đoán tự động (`diagnosis`) | _<INTENDED / LIKELIHOOD DISPLACEMENT / FAILURE / AMBIGUOUS>_ |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | _<... → ... ký tự>_ |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

_Trả lời ở đây._

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | | | | | | | |
| hữu ích — helpfulness (4) | | | | | | | |
| an toàn — safety (4) | | | | | | | |

Giám khảo: ______ · sanity accuracy: ______ · `score_length_spearman` (reward model) hoặc độ nhất quán khi đổi chỗ A/B — position consistency (giám khảo API): ______

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

_Trả lời ở đây._

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

_Trả lời ở đây._

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
