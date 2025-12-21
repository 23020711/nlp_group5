# nlp_group5
## 1. Dữ liệu & Pipeline
- Dữ liệu: IWSLT'15 (Vietnamese - English).
- Code xử lý dữ liệu và training pipeline nằm trong file `NhomX_Training_Pipeline.ipynb`.
- Biểu đồ Loss huấn luyện: `loss_graph.png`.

## 2. Model Checkpoint
file trọng số mô hình đã huấn luyện (`transformer_best.pt`) được lưu trữ tại link sau: https://drive.google.com/drive/folders/1T4Ci8E-VG67_BnBtdC8kXaKTXjtzuZeF?usp=drive_link

## 3. Cách chạy Demo
1. Tải file model từ link trên và để vào thư mục `checkpoints/`.
2. Chạy notebook `Inference.ipynb`.
## 4. VLSP 2025 – Dịch Máy Tiếng Anh ↔ Tiếng Việt Lĩnh vực Y khoa
**Fine-tune Qwen2-1.5B-Instruct trên dữ liệu MedEV/nhuvo đã lọc theo từ khóa y tế**
1. Tải bộ dữ liệu song ngữ EN-VI từ MedEV 
2. Fine-tune **Qwen/Qwen2-1.5B-Instruct** bằng LoRA + 4-bit quantization (tiết kiệm VRAM, chạy được trên Colab T4 16 GB)  
3. Đánh giá bằng **BLEU**, **ROUGE-L**, vẽ biểu đồ BLEU từng câu và phân tích lỗi dịch chi tiết
## Cách chạy nhanh (Google Colab)
https://colab.research.google.com/drive/1WzqM2EYKRfq0bLLFrP6c7GPOWCANlnKP?authuser=1#scrollTo=k1BfqJZ_VFg7
1. Mở link Colab ở trên  
2. Runtime → Change runtime type → GPU (T4 hoặc tốt hơn)  
3. Chạy lần lượt từng cell (Shift + Enter)
