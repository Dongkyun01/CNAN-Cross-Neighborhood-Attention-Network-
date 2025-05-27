# train.py (사전 계산 저장 + 샤딩 구조 적용 + 메모리 안정성 향상)

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


def train(model, train_data, precomputed_neighbors, batch_size=128, epochs=10, lr=0.001, log_path="training_log.csv"):
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

                S1 = S1.to(device)
                T1 = T1.to(device)
                S2 = S2.to(device)
                T2 = T2.to(device)
                rating = r.unsqueeze(0)

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
    print("[INFO] 모델 저장 완료: cnan_model.pt")


def cosine_similarity(a, b):
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return torch.dot(a, b).item()


def get_2hop_users(user_id, user_to_items, item_to_users, model, top_k=2):
    items = user_to_items.get(user_id, [])
    candidates = set()
    for item in items:
        candidates.update(item_to_users.get(item, []))
    candidates.discard(user_id)

    target_emb = model.user_embeddings(torch.tensor(user_id, device=next(model.parameters()).device))
    sims = [(u, cosine_similarity(target_emb, model.user_embeddings(torch.tensor(u, device=target_emb.device)))) for u in candidates]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in sims[:top_k]]


def get_2hop_items(item_id, user_to_items, item_to_users, model, top_k=2):
    users = item_to_users.get(item_id, [])
    candidates = set()
    for user in users:
        candidates.update(user_to_items.get(user, []))
    candidates.discard(item_id)

    target_emb = model.item_embeddings(torch.tensor(item_id, device=next(model.parameters()).device))
    sims = [(i, cosine_similarity(target_emb, model.item_embeddings(torch.tensor(i, device=target_emb.device)))) for i in candidates]
    sims.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in sims[:top_k]]


def precompute_all_neighbors(model, train_data, user_to_items, item_to_users, top_k=2, save_dir="precomputed_chunks", chunk_size=10000):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    embedding_dim = model.user_embeddings.embedding_dim
    precomputed = {}

    shard_id = 0
    chunk = {}

    progress_bar = tqdm(total=len(train_data), desc="Precomputing")
    for idx, row in enumerate(train_data):
        user_id = int(row[0].item())
        item_id = int(row[1].item())
        key = (user_id, item_id)

        user_1hop_items = user_to_items.get(user_id, [])
        item_1hop_users = item_to_users.get(item_id, [])

        S1 = torch.stack([model.item_embeddings(torch.tensor(i, device=device)) for i in user_1hop_items]) \
            if user_1hop_items else torch.zeros((1, embedding_dim), device=device)
        T1 = torch.stack([model.user_embeddings(torch.tensor(u, device=device)) for u in item_1hop_users]) \
            if item_1hop_users else torch.zeros((1, embedding_dim), device=device)

        user_2hop_users = get_2hop_users(user_id, user_to_items, item_to_users, model, top_k)
        item_2hop_items = get_2hop_items(item_id, user_to_items, item_to_users, model, top_k)

        S2 = torch.stack([model.user_embeddings(torch.tensor(u, device=device)) for u in user_2hop_users]) \
            if user_2hop_users else torch.zeros((1, embedding_dim), device=device)
        T2 = torch.stack([model.item_embeddings(torch.tensor(i, device=device)) for i in item_2hop_items]) \
            if item_2hop_items else torch.zeros((1, embedding_dim), device=device)

        chunk[key] = (S1.detach().cpu(), T1.detach().cpu(), S2.detach().cpu(), T2.detach().cpu())
        progress_bar.update(1)

        if len(chunk) >= chunk_size:
            path = os.path.join(save_dir, f"shard_{shard_id}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(chunk, f)
            print(f"[INFO] 저장: {path} ({len(chunk)}개) 완료")
            shard_id += 1
            chunk.clear()

    # 마지막 남은 데이터 저장
    if chunk:
        path = os.path.join(save_dir, f"shard_{shard_id}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(chunk, f)
        print(f"[INFO] 저장: {path} ({len(chunk)}개) 완료")

    progress_bar.close()
    print("[INFO] 전체 샤딩 사전 계산 완료")

    # 모든 shard 로드하여 통합 반환 (추후 메모리 최적화를 위해 optional로 분기 가능)
    all_precomputed = {}
    for filename in os.listdir(save_dir):
        if filename.endswith(".pkl"):
            with open(os.path.join(save_dir, filename), 'rb') as f:
                all_precomputed.update(pickle.load(f))

    return all_precomputed


if __name__ == '__main__':
    data = load_movielens_100k(r"C:\Users\USER\Desktop\추천시스템연습\ml-100k\u.data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNAN(data['num_users'], data['num_items'], embedding_dim=32).to(device)
    precomputed_neighbors = precompute_all_neighbors(model, data['train_tensor'], data['user_to_items'], data['item_to_users'])

    train(model, data['train_tensor'], precomputed_neighbors, epochs=10, lr=0.001)



