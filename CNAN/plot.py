import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import matplotlib
import seaborn as sns
matplotlib.use('Agg')


def plot_pred_vs_true(csv_path="eval_predictions.csv"):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6, 6))
    plt.scatter(df['true'], df['pred'], alpha=0.5, edgecolors='k')
    plt.plot([df['true'].min(), df['true'].max()], [df['true'].min(), df['true'].max()], 'r--')
    plt.xlabel("True Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Predicted vs True Ratings")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_pred_vs_true.png")
    plt.close()
    print("[✅ 저장 완료] scatter_pred_vs_true.png")

def plot_error_distribution(csv_path="eval_predictions.csv"):
    df = pd.read_csv(csv_path)
    errors = df['pred'] - df['true']
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Prediction Error (pred - true)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("error_histogram.png")
    plt.close()
    print("[✅ 저장 완료] error_histogram.png")



def plot_user_item_error_heatmap(csv_path="eval_predictions.csv", max_users=50, max_items=50):
    df = pd.read_csv(csv_path)
    df['abs_error'] = abs(df['pred'] - df['true'])

    # 피벗테이블로 user-item 행렬 생성
    error_matrix = df.pivot(index='user', columns='item', values='abs_error')
    error_matrix = error_matrix.fillna(0).clip(upper=5)[:max_users].iloc[:, :max_items]  # 일부만 보기 좋게 자름

    plt.figure(figsize=(10, 8))
    sns.heatmap(error_matrix, cmap='YlOrRd', linewidths=0.5)
    plt.title("User-Item Absolute Error Heatmap")
    plt.xlabel("Item ID")
    plt.ylabel("User ID")
    plt.tight_layout()
    plt.savefig("user_item_error_heatmap.png")
    plt.close()
    print("[✅ 저장 완료] user_item_error_heatmap.png")

def plot_loss_and_lr_from_csv(csv_path="training_log.csv"):
    epochs, losses, lrs = [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['avg_loss']))
            lrs.append(float(row['lr']))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, color=color, label='Loss', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(epochs, lrs, color=color, label='Learning Rate', marker='x')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Loss and Learning Rate per Epoch")

    save_path = os.path.join(os.getcwd(), "loss_lr_curve.png")
    plt.savefig(save_path)
    print(f"[✅ 저장 완료] {save_path}")
    plt.close()

def plot_eval_metrics_from_csv(csv_path="eval_log.csv"):
    rmse, mae = None, None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rmse = float(row["rmse"])
            mae = float(row["mae"])

    if rmse is None or mae is None:
        print("[❌ 오류] eval_log.csv 데이터가 잘못되었습니다.")
        return

    # 단일 바 차트 출력
    plt.figure(figsize=(6, 4))
    plt.bar(["RMSE", "MAE"], [rmse, mae], color=["skyblue", "salmon"])
    plt.title("Evaluation Metrics (RMSE / MAE)")
    plt.ylabel("Score")
    plt.grid(axis="y")
    save_path = os.path.join(os.getcwd(), "eval_bar_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✅ 저장 완료] {save_path}")
    plt.close()

if __name__ == '__main__':
    plot_pred_vs_true("eval_predictions.csv")
    plot_error_distribution("eval_predictions.csv")
    plot_user_item_error_heatmap("eval_predictions.csv")
    plot_loss_and_lr_from_csv()
    plot_eval_metrics_from_csv()