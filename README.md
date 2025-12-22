#  Hệ thống Dịch máy Anh-Việt: Từ Transformer từ đầu đến Fine-tuning LLM

Dự án này thực hiện hai bài toán chính:
1. **Xây dựng mô hình Transformer từ đầu** (Problem 1) cho dịch Anh-Việt qua 6 giai đoạn (phases) với các cải tiến liên tục.
2. **Fine-tuning mô hình ngôn ngữ lớn (LLM) Qwen2.5-1.5B** (Problem 2) cho dịch thuật chuyên ngành Y tế bằng kỹ thuật QLoRA.

---

##  Cấu trúc thư mục

```
NLP/
├── Report.pdf           # Report project
├── .gitignore
├── README.md
├── Problem1/
│   ├── phase1/          # Baseline với word-level tokenization
│   ├── phase2/          # Cải thiện preprocessing + Label Smoothing
│   ├── phase3/          # Thử nghiệm BPE (model sập, chưa thành công)
│   ├── phase4/          # Tối ưu kiến trúc + BPE + Noam Scheduler (model sập)
│   ├── phase5/          # Beam Search + BPE
│   └── phase6/          # Best version: BPE + Noam + Beam Search + training dài
│       ├── training_inference_evaluate_pipeline.ipynb
│       ├── inference_evaluate.ipynb
│       ├── loss_graph.png
│       ├── model.txt          # Link tải model
│       └── result.txt         # Kết quả BLEU và ví dụ dịch
├── Problem2/
│   ├── Train.ipynb            # Huấn luyện Qwen2.5 với QLoRA
│   ├── evaluate.ipynb         # Đánh giá trên tập test y tế
│   ├── massive_eval_results.csv               # Chi tiết kết quả những câu đã dịch
│   ├── translation_comparison_report.png      # Báo cáo So sánh 2 chiều dịch
│   └── model.txt              # Link tải model fine-tuned
└── transformer_from_scratch/
    ├── transformer.py         # Kiến trúc Transformer từ đầu
    ├── components.py          # Các thành phần (Attention, FFN, ...)
    └── Transformer.pdf        # Báo cáo kiến trúc
```

---

##  Yêu cầu hệ thống & Cài đặt

### Problem 1 (Phase 1–6):
- Chạy trên **Google Colab** (khuyến nghị)
- GPU: T4 hoặc V100 (miễn phí trên Colab)
- Thư viện chính: `torch`, `pyvi`, `spacy`, `sacrebleu`

### Problem 2:
- Chạy trên **Kaggle** (hoặc Colab Pro với GPU lớn)
- GPU: T4 hoặc P100 (trên Kaggle)
- Thư viện: `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`

---

##  Hướng dẫn chạy thử

###  **Problem 1 – Phase 6 (Best Version)**
**Môi trường:** Google Colab  
**Dữ liệu:** Tự động tải từ [https://github.com/stefan-it/nmt-en-vi](https://github.com/stefan-it/nmt-en-vi)

1. Mở file `training_inference_evaluate_pipeline.ipynb` trong thư mục `Problem1/phase6/`.
2. Chạy tuần tự các cell:
   - Cài đặt thư viện (`pyvi`, `spacy`)
   - Tải và làm sạch dữ liệu
   - Xây dựng vocab và dataloader
   - Khởi tạo model Transformer (từ file `transformer.py`)
   - Huấn luyện model (có thể load checkpoint từ `model.txt`)
   - Dịch thử với Beam Search
   - Đánh giá bằng BLEU trên tập test IWSLT 2013
3. Kết quả sẽ lưu trong `result.txt` và hình ảnh loss trong `loss_graph.png`.

**Chú ý:** Nếu muốn chạy nhanh, có thể bỏ qua training và load trực tiếp model từ link trong `model.txt`.

---

###  **Problem 2 – Fine-tuning Qwen2.5 cho dịch Y tế**
**Môi trường:** Kaggle Notebook  
**Dữ liệu:** Tập VLSP 2025 (yêu cầu truy cập từ canvas UET)

1. Upload dữ liệu `train.vi.txt`, `train.en.txt`, `public_test.vi.txt`, `public_test.en.txt` lên Kaggle.
2. Mở notebook `Train.ipynb`:
   - Cài đặt thư viện (`peft`, `bitsandbytes`, `trl`)
   - Load dataset và tokenizer Qwen2.5-1.5B
   - Cấu hình QLoRA (rank=32, alpha=64)
   - Huấn luyện với `SFTTrainer`
   - Lưu model fine-tuned
3. Đánh giá bằng notebook `evaluate.ipynb`:
   - Load model đã fine-tune
   - Dịch 1000 mẫu test với prompt chuẩn
   - Tính BLEU, TER, Semantic Score (BERTScore)
   - Xuất báo cáo và biểu đồ so sánh
4. Link model kaggle được lưu ở file model.txt nếu muốn sử dụng.

**LƯU Ý: NÊN SỬ DỤNG PROMPT TRONG file evaluate.ipynb và xử lý EARLY STOP response như trong file đó để đưa ra kết quả tốt nhất, TÔI ĐÃ THỬ NHỮNG PROMPT KHÁC NHƯNG MODEL BỊ ẢO GIÁC VÀ CỐ GEN HẾT TỪ TRONG TOKEN CHO PHÉP**

**Kết quả mẫu:**
- BLEU (En→Vi): 32.53
- BLEU (Vi→En): 23.52
- Semantic Score: 0.8927

---

##  Kết quả chính

### Problem 1 – Transformer từ đầu:
- **Phase 6 (best):** BLEU = 16.18 trên IWSLT 2013
- Cải tiến qua từng phase: từ 11.69 (phase 1) lên 16.18 (phase 6)
- Kỹ thuật áp dụng: BPE, Noam Scheduler, Beam Search, Label Smoothing

### Problem 2 – LLM fine-tuning:
- **Qwen2.5-1.5B + QLoRA:** BLEU = 32.53 (En→Vi) trên tập y tế
- Hiệu quả vượt trội so với Transformer tự xây dựng
- Tối ưu bộ nhớ với quantization 4-bit

---

##  Tài liệu tham khảo

- [Transformer: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Qwen2.5 Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- [VLSP 2025 Shared Task](https://vlsp.org.vn/)

---

##  Đóng góp

Nhóm sinh viên UET – Đại học Công nghệ, ĐHQGHN:
- Vũ Văn Phong (22028309)
- Đinh Huyền Trang (23020711)
- Triệu Việt Hùng (22028069)

**GitHub:** [https://github.com/23020711/nlp_group5](https://github.com/23020711/nlp_group5)

---

##  Giấy phép

Dự án được phát triển cho mục đích học thuật, mã nguồn mở theo MIT License.