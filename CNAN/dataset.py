# dataset.py

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict

def load_movielens_100k(data_path=r"C:\Users\USER\Desktop\추천시스템연습\ml-100k\u.data", device=None):
    """
    MovieLens 100K 데이터셋을 로드하고, 학습/테스트 데이터를 전처리합니다.

    Args:
        data_path (str): u.data 파일 경로
        device (torch.device): 'cuda' 또는 'cpu'

    Returns:
        dict: {
            'train_tensor': 학습 데이터 (user, item, rating),
            'test_tensor': 테스트 데이터 (user, item, rating),
            'user_to_items': 유저-아이템 관계 (dict),
            'item_to_users': 아이템-유저 관계 (dict),
            'num_users': 유저 수,
            'num_items': 아이템 수
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 데이터 로드
    df = pd.read_csv(data_path, sep="\t", names=["user", "item", "rating", "timestamp"])
    df.drop("timestamp", axis=1, inplace=True)

    # 2. user/item ID를 0부터 시작하도록 조정
    df["user"] -= 1
    df["item"] -= 1

    num_users = df["user"].nunique()
    num_items = df["item"].nunique()

    # 3. 학습 / 테스트 분할
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 4. 1-hop 관계 구성 (dictionary)
    user_to_items = defaultdict(list)
    item_to_users = defaultdict(list)

    for row in train_df.itertuples():
        user_to_items[row.user].append(row.item)
        item_to_users[row.item].append(row.user)

    # 5. tensor 변환 (float32 + GPU 이동)
    train_tensor = torch.tensor(train_df[["user", "item", "rating"]].values,
                                dtype=torch.float32, device=device)
    test_tensor = torch.tensor(test_df[["user", "item", "rating"]].values,
                               dtype=torch.float32, device=device)

    # 6. 결과 반환
    return {
        "train_tensor": train_tensor,
        "test_tensor": test_tensor,
        "user_to_items": user_to_items,
        "item_to_users": item_to_users,
        "num_users": num_users,
        "num_items": num_items
    }

