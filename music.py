import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix
import xgboost as xgb

# Load data
train = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/102/train.csv")
test = pd.read_csv("https://s3.amazonaws.com/hackerday.datascience/102/test.csv")

# Create composite key
train["key"] = train["msno"] + "_" + train["song_id"]
test["key"] = test["msno"] + "_" + test["song_id"]

# Assign IDs and types
train["id"] = train.index.astype(str)
test["target"] = ''
train["type"] = "train"
test["type"] = "test"

# Subset relevant columns
cols = ['key', 'type', 'id', 'source_system_tab', 'source_screen_name', 'source_type', 'target']
train1 = train[cols]
test1 = test[cols]

# Combine train and test
master_df = pd.concat([train1, test1], axis=0).reset_index(drop=True)

# One-hot feature engineering (source_system_tab)
for tab in master_df['source_system_tab'].dropna().unique():
    master_df[f'flag_source_system_tab_{tab.replace(" ", "_")}'] = (master_df['source_system_tab'] == tab).astype(int)

# One-hot feature engineering (source_type)
for typ in master_df['source_type'].dropna().unique():
    master_df[f'flag_source_type_{typ.replace("-", "_").replace(" ", "_")}'] = (master_df['source_type'] == typ).astype(int)

# Feature interactions
def safe_get(col):
    return master_df[col] if col in master_df.columns else 0

master_df['flag_source_type_system_tab_artist_my_library'] = safe_get('flag_source_type_artist') * safe_get('flag_source_system_tab_my_library')
master_df['flag_source_type_system_tab_listen_with'] = safe_get('flag_source_system_tab_listen_with') * safe_get('flag_source_type_listen_with')
master_df['flag_source_type_system_tab_local_library_my_library'] = safe_get('flag_source_type_local_library') * safe_get('flag_source_system_tab_my_library')
master_df['flag_source_type_system_tab_local_library_discover'] = safe_get('flag_source_type_local_library') * safe_get('flag_source_system_tab_discover')
master_df['flag_source_type_system_tab_local_playlist_my_library'] = safe_get('flag_source_type_local_playlist') * safe_get('flag_source_system_tab_my_library')
master_df['flag_source_type_system_tab_online_playlist_discover'] = safe_get('flag_source_type_online_playlist') * safe_get('flag_source_system_tab_discover')
master_df['flag_source_type_system_tab_song_search'] = safe_get('flag_source_type_song') * safe_get('flag_source_system_tab_search')
master_df['flag_source_type_system_tab_song_based_playlist_discover'] = safe_get('flag_source_type_song_based_playlist') * safe_get('flag_source_system_tab_discover')

# Split train/test
train_data = master_df[master_df["type"] == "train"].copy()
test_data = master_df[master_df["type"] == "test"].copy()

# Prepare features (start from column 7 based on R code; but dynamically handled here)
feature_cols = [col for col in master_df.columns if col.startswith("flag_")]
X_train = csr_matrix(train_data[feature_cols].values)
X_test = csr_matrix(test_data[feature_cols].values)

y_train = train_data["target"].astype(int)

# Set XGBoost parameters
params = {
    "objective": "multi:softprob",
    "num_class": 2,
    "eval_metric": "merror",
    "nthread": 8,
    "max_depth": 16,
    "eta": 0.3,
    "gamma": 0,
    "subsample": 1,
    "colsample_bytree": 1,
    "min_child_weight": 12
}

# Cross-validation
dtrain = xgb.DMatrix(X_train, label=y_train)
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=30,
    nfold=2,
    early_stopping_rounds=5,
    verbose_eval=True,
    seed=42
)

# Train final model
best_rounds = len(cv_results)
model = xgb.train(params, dtrain, num_boost_round=best_rounds)

# Predict
dtest = xgb.DMatrix(X_test)
preds = model.predict(dtest)

# Convert predictions to labels
pred_labels = np.where(preds[:, 0] > preds[:, 1], 0, 1)

# Prepare output
output = pd.DataFrame({
    "id": test_data["id"],
    "target": pred_labels
})

# Save to CSV
output.to_csv("predictions.csv", index=False)
