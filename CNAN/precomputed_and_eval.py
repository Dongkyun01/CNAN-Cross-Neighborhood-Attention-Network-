# eval.py (사전 계산 기반 평가 + 샤딩 저장)
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from model import CNAN
from dataset import load_movielens_100k
from precomputed_and_train import precompute_all_neighbors
import os
import csv
import pickle

def load_precomputed_neighbors(save_dir):
    precomputed = {}
    for fname in os.listdir(save_dir):
        if fname.endswith(".pkl"):
            with open(os.path.join(save_dir, fname), 'rb') as f:
                precomputed.update(pickle.load(f))
    print(f"[INFO] 총 {len(precomputed)}개 테스트 이웃 임베딩 불러옴")
    return precomputed

def evaluate(model, test_data, precomputed_neighbors, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for row in tqdm(test_data, desc="Evaluating"):
            user_id = int(row[0].item())
            item_id = int(row[1].item())
            true_rating = row[2].item()

            S1, T1, S2, T2 = precomputed_neighbors[(user_id, item_id)]
            S1 = S1.to(device)
            T1 = T1.to(device)
            S2 = S2.to(device)
            T2 = T2.to(device)

            pred = model(S1, T1, S2, T2).item()
            preds.append(pred)
            trues.append(true_rating)

    preds = np.array(preds)
    trues = np.array(trues)

    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    mae = np.mean(np.abs(preds - trues))

    print(f"\nEvaluation Results:\nRMSE: {rmse:.4f}\nMAE:  {mae:.4f}")

    # CSV 저장
    with open("eval_log.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rmse", "mae"])
        writer.writerow([rmse, mae])
    # save per-user prediction for later analysis
    with open("eval_predictions.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user", "item", "true", "pred"])
        for i in range(len(preds)):
            writer.writerow([int(test_data[i][0].item()), int(test_data[i][1].item()), trues[i], preds[i]])

    return rmse, mae

if __name__ == '__main__':
    data = load_movielens_100k(r"C:\Users\USER\Desktop\추천시스템연습\ml-100k\u.data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNAN(data['num_users'], data['num_items'], embedding_dim=32).to(device)
    model.load_state_dict(torch.load("cnan_model.pt"))

    # 저장된 사전 계산 파일이 있으면 불러오기, 없으면 새로 계산
    save_dir = "precomputed_test_chunks"
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        precomputed_neighbors = load_precomputed_neighbors(save_dir)
    else:
        print("[INFO] Precomputing test neighbors...")
        precomputed_neighbors = precompute_all_neighbors(
            model, data['test_tensor'], data['user_to_items'], data['item_to_users'],
            save_dir=save_dir
        )

    evaluate(model, data['test_tensor'], precomputed_neighbors, device)
