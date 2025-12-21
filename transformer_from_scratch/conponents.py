import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Tạo ma trận [max_len, d_model] chứa thông tin vị trí
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Áp dụng công thức Sinusoidal
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Thêm chiều batch: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Đăng ký buffer (không phải là tham số huấn luyện)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # Cộng embedding của từ với positional encoding tương ứng
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model phải chia hết cho n_head"

        self.d_head = d_model // n_head
        self.n_head = n_head
        self.d_model = d_model

        # Các lớp Linear để chiếu Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Lớp Linear cuối cùng sau khi nối các head
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # 1. Tính Q, K, V qua các lớp Linear
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. Chia thành nhiều đầu (Split heads)
        # Biến đổi: [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_head, d_head]
        # Sau đó transpose để đưa n_head lên trước: [batch_size, n_head, seq_len, d_head]
        Q = Q.view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention
        # energy: [batch_size, n_head, seq_len_q, seq_len_k]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_head)

        # Áp dụng Mask (nếu có) - Dùng cho Decoder (Look-ahead mask) hoặc Padding mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(energy, dim=-1)

        # x: [batch_size, n_head, seq_len_q, d_head]
        x = torch.matmul(attention, V)

        # 4. Nối các đầu lại (Concat)
        # [batch_size, seq_len_q, n_head * d_head] = [batch_size, seq_len_q, d_model]
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        return self.fc_out(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model phải chia hết cho n_head"

        self.d_head = d_model // n_head
        self.n_head = n_head
        self.d_model = d_model

        # Các lớp Linear để chiếu Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Lớp Linear cuối cùng sau khi nối các head
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # 1. Tính Q, K, V qua các lớp Linear
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2. Chia thành nhiều đầu (Split heads)
        # Biến đổi: [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_head, d_head]
        # Sau đó transpose để đưa n_head lên trước: [batch_size, n_head, seq_len, d_head]
        Q = Q.view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention
        # energy: [batch_size, n_head, seq_len_q, seq_len_k]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_head)

        # Áp dụng Mask (nếu có) - Dùng cho Decoder (Look-ahead mask) hoặc Padding mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(energy, dim=-1)

        # x: [batch_size, n_head, seq_len_q, d_head]
        x = torch.matmul(attention, V)

        # 4. Nối các đầu lại (Concat)
        # [batch_size, seq_len_q, n_head * d_head] = [batch_size, seq_len_q, d_model]
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        return self.fc_out(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


