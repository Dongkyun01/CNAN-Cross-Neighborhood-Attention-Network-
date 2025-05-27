# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNAN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(CNAN, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # 학습 가능한 projection matrix P1, P2
        self.P1 = nn.Parameter(torch.eye(embedding_dim))
        self.P2 = nn.Parameter(torch.eye(embedding_dim))

        # CNN 블록: attention matrix 처리용
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2),  # 입력채널 1, 출력채널 4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fully Connected layer → 최종 평점 예측
        self.fc = nn.Linear(4, 1)

    def forward(self, S1, T1, S2, T2):
        A1 = S1 @ self.P1 @ T1.T
        A2 = S2 @ self.P2 @ T2.T

        def process_attention(att):
            if att.size(-2) < 2 or att.size(-1) < 2:
                return torch.zeros((1, 4), device=att.device)
            x = att.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            out = self.cnn(x)
            return out.view(1, -1)  # (1, 4)

        a1 = process_attention(A1)
        a2 = process_attention(A2)

        out = self.fc(a1 + a2)
        return out.view(-1)  # 스칼라 형태로 출력


