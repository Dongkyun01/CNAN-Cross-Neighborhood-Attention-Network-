# eval_cached.py — 저장된 precomputed_test_chunks 기반 평가

import torch
import numpy as np
import csv
from tqdm import tqdm
from model import CNAN
from dataset import load_movielens_100k
import pickle
import os

def load_precomputed_neighbors(save_dir):
    precomputed = {}
    for fname in os.listdir(save_dir):
        if fname.endswith(".pkl"):
            with open(os.path.join(save_dir, fname), 'rb') as f:
                precomputed.update(pickle.load(f))
    print(f"[INFO] 사전 계산된 테스트 이웃 {len(precomputed)}개 로드됨")
    return precomputed

def evaluate(model, test_data, precomputed_neighbors, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for row in tqdm(test_data, desc="Evaluating (Cached)"):
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

    print(f"\n[결과] RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # 평가 결과 로그 저장
    with open("eval_log.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rmse", "mae"])
        writer.writerow([rmse, mae])

    return rmse, mae


if __name__ == '__main__':
    # 1. 데이터 로드
    data = load_movielens_100k(r"C:\Users\USER\Desktop\추천시스템연습\ml-100k\u.data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 모델 로딩
    model = CNAN(data['num_users'], data['num_items'], embedding_dim=32).to(device)
    model.load_state_dict(torch.load("cnan_model.pt"))

    # 3. 사전계산된 이웃 정보 불러오기
    cached_dir = "precomputed_test_chunks"
    if not os.path.exists(cached_dir):
        raise FileNotFoundError(f"{cached_dir} 디렉토리가 존재하지 않습니다. 먼저 eval.py로 사전 계산을 수행하세요.")

    precomputed_neighbors = load_precomputed_neighbors(cached_dir)

    # 4. 평가 실행
    evaluate(model, data['test_tensor'], precomputed_neighbors, device)
