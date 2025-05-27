# retrain_with_cache.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model import CNAN
from dataset import load_movielens_100k
from tqdm import tqdm
import os
import csv
import pickle


def train_with_cached_neighbors(model, train_data, precomputed_neighbors, batch_size=128, epochs=10, lr=0.001, log_path="training_log.csv"):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(
        train_data[:, 0].long(),
        train_data[:, 1].long(),
        train_data[:, 2].float()
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if os.path.exists(log_path):
        os.remove(log_path)
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "lr"])

    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False)
        for user_ids, item_ids, ratings in loop:
            batch_loss = 0.0
            for u, i, r in zip(user_ids, item_ids, ratings):
                key = (int(u.item()), int(i.item()))
                S1, T1, S2, T2 = precomputed_neighbors.get(key, (None, None, None, None))
                if S1 is None:
                    continue

                # Move to device
                S1 = S1.to(device)
                T1 = T1.to(device)
                S2 = S2.to(device)
                T2 = T2.to(device)
                rating = r.unsqueeze(0).to(device)

                pred = model(S1, T1, S2, T2)
                loss = loss_fn(pred, rating)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            total_loss += batch_loss
            loop.set_postfix(batch_loss=batch_loss, lr=scheduler.get_last_lr()[0])

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, current_lr])

    torch.save(model.state_dict(), "cnan_model.pt")
    print("[INFO] 수정한 모델 저장 완료: cnan_model.pt")


if __name__ == '__main__':
    data = load_movielens_100k(r"C:\Users\USER\Desktop\추천시스템연습\ml-100k\u.data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNAN(data['num_users'], data['num_items'], embedding_dim=32).to(device)

    # Precomputed shard load
    precomputed_neighbors = {}
    shard_dir = "precomputed_chunks"
    for filename in os.listdir(shard_dir):
        if filename.endswith(".pkl"):
            with open(os.path.join(shard_dir, filename), 'rb') as f:
                precomputed_neighbors.update(pickle.load(f))
    print(f"[INFO] 사전 계산 데이터 로드 완료: {len(precomputed_neighbors)} 개")

    train_with_cached_neighbors(model, data['train_tensor'], precomputed_neighbors, epochs=10, lr=0.001)
