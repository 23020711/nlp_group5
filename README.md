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