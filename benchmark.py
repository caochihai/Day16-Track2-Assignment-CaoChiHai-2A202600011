import pandas as pd
import lightgbm as lgb
import time
import json
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import make_classification
from tqdm import tqdm

print("1. Đang tạo dữ liệu  (1 Triệu dòng, 40 Features, có nhiễu)...")
start_time = time.time()
X, y = make_classification(
    n_samples=1000000, 
    n_features=40, 
    n_informative=20, 
    n_redundant=10, 
    n_classes=2, 
    weights=[0.95, 0.05], 
    class_sep=0.7,
    flip_y=0.05, 
    random_state=42
)
load_time = time.time() - start_time

print("2. Đang chia tập dữ liệu (Test = 20%, Train+Dev = 80%)...")
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Bắt đầu huấn luyện 5-Fold Cross Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'objective': 'binary', 
    'metric': 'auc', 
    'verbose': -1, 
    'num_threads': 8,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 63
}

total_boost_rounds = 1000
train_auc_list, dev_auc_list = [], []
train_f1_list, dev_f1_list = [], []
models = []
train_start_time = time.time()

# Hàm Custom Callback để LightGBM "giao tiếp" với tqdm
def tqdm_callback(pbar):
    def callback(env):
        pbar.update(1)
        # Hiển thị luôn điểm AUC mới nhất lên góc phải thanh tiến trình
        if env.evaluation_result_list:
            score = env.evaluation_result_list[0][2]
            pbar.set_postfix({'Dev AUC': f'{score:.4f}'})
    return callback

for fold, (train_idx, dev_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    print(f"\n---> BẮT ĐẦU FOLD {fold + 1}/5:")
    X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
    X_dev, y_dev = X_train_full[dev_idx], y_train_full[dev_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    dev_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data)
    
    # Tạo thanh tiến trình tqdm cho mỗi Fold
    with tqdm(total=total_boost_rounds, desc=f"Traning Fold {fold+1}", bar_format="{l_bar}{bar:40}{r_bar}") as pbar:
        model = lgb.train(
            params, 
            train_data, 
            num_boost_round=total_boost_rounds,
            valid_sets=[train_data, dev_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                tqdm_callback(pbar)
            ]
        )
    models.append(model)
    
    # Dự đoán
    tr_pred_prob = model.predict(X_tr)
    dev_pred_prob = model.predict(X_dev)
    
    # Tính điểm
    train_auc_list.append(roc_auc_score(y_tr, tr_pred_prob))
    dev_auc_list.append(roc_auc_score(y_dev, dev_pred_prob))
    train_f1_list.append(f1_score(y_tr, (tr_pred_prob > 0.5).astype(int)))
    dev_f1_list.append(f1_score(y_dev, (dev_pred_prob > 0.5).astype(int)))
    
    print(f"[*] Hoàn thành Fold {fold + 1} | Dev AUC: {dev_auc_list[-1]:.4f} | Train AUC: {train_auc_list[-1]:.4f}")

train_time = time.time() - train_start_time

print("\n4. Đang đánh giá Ensemble trên tập Test...")
test_preds = np.zeros(len(X_test))
for model in models:
    test_preds += model.predict(X_test) / len(models)

y_pred = (test_preds > 0.5).astype(int)

test_auc = roc_auc_score(y_test, test_preds)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_prec = precision_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)

print("\n" + "="*50)
print("     BÁO CÁO KẾT QUẢ BENCHMARK ")
print("="*50)
print(f"Cấu hình máy ảo     : n2-standard-8 (8 vCPUs)")
print(f"Tổng số dữ liệu     : 1.000.000 dòng x 40 Cột")
print(f"Thời gian sinh data : {load_time:.2f} giây")
print(f"Thời gian Training  : {train_time:.2f} giây")
print("-" * 50)
print(f"[TRUNG BÌNH 5-FOLD CV TRÊN TẬP TRAIN & DEV]")
print(f"Train AUC-ROC       : {np.mean(train_auc_list):.4f}")
print(f"Train F1-Score      : {np.mean(train_f1_list):.4f}")
print(f"Dev AUC-ROC         : {np.mean(dev_auc_list):.4f}")
print(f"Dev F1-Score        : {np.mean(dev_f1_list):.4f}")
print("-" * 50)
print(f"[ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (20%)]")
print(f"Test AUC-ROC        : {test_auc:.4f}")
print(f"Test Accuracy       : {test_acc:.4f}")
print(f"Test F1-Score       : {test_f1:.4f}")
print(f"Test Precision      : {test_prec:.4f}")
print(f"Test Recall         : {test_rec:.4f}")
print("="*50)

metrics = {
    "train_time_sec": round(train_time, 2),
    "mean_dev_auc": round(np.mean(dev_auc_list), 4),
    "test_auc": round(test_auc, 4),
    "test_f1": round(test_f1, 4)
}
with open('benchmark_result.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("\nĐã lưu kết quả vào file benchmark_result.json")
