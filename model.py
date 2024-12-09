import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from geopy.distance import great_circle

# --- Load Data ---
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# --- Define Target and ID Columns ---
target = 'is_fraud'
id_col = 'id'


# --- Feature Engineering ---
def process_datetime_features(df):
    """Extract features from transaction date and time."""
    df['datetime'] = pd.to_datetime(df['trans_date'] + ' ' + df['trans_time'], 
                                    format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    return df


def process_dob(df):
    """Calculate age from date of birth."""
    df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d', errors='coerce')
    df['age'] = df.apply(lambda x: x['year'] - x['dob'].year if pd.notnull(x['dob']) else np.nan, axis=1)
    return df


def process_distance(df):
    """Calculate distance between cardholder and merchant locations."""
    def calc_dist(row):
        if pd.notnull(row['lat']) and pd.notnull(row['long']) and pd.notnull(row['merch_lat']) and pd.notnull(row['merch_long']):
            return great_circle((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km
        else:
            return np.nan
    df['distance'] = df.apply(calc_dist, axis=1)
    return df


def process_amount(df):
    """Apply log transformation to the transaction amount."""
    df['amt_log'] = np.log1p(df['amt'])
    return df


def process_categorical(df):
    """Convert categorical columns to string type."""
    cat_cols = ['category', 'gender', 'state', 'job', 'city', 'merchant']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def encode_categoricals(train, test, cols):
    """Label encode categorical features."""
    for c in cols:
        le = LabelEncoder()
        combined = pd.concat([train[c], test[c]], axis=0).astype(str)
        le.fit(combined)
        train[c] = le.transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))
    return train, test


def feature_engineering(train, test):
    """Perform feature engineering on training and testing datasets."""
    train = process_datetime_features(train)
    test = process_datetime_features(test)

    train = process_dob(train)
    test = process_dob(test)

    train = process_distance(train)
    test = process_distance(test)

    train = process_amount(train)
    test = process_amount(test)

    train = process_categorical(train)
    test = process_categorical(test)

    cat_features = ['category', 'gender', 'state', 'job', 'city', 'merchant']
    train, test = encode_categoricals(train, test, cat_features)

    # New Feature: Speed of transaction based on distance and transaction time
    train['transaction_speed'] = train['distance'] / (train['amt'] + 1)
    test['transaction_speed'] = test['distance'] / (test['amt'] + 1)

    # New Feature: Distance per unit transaction amount
    train['distance_per_amount'] = train['distance'] / (train['amt'] + 1)
    test['distance_per_amount'] = test['distance'] / (test['amt'] + 1)

    return train, test


train, test = feature_engineering(train, test)

# --- Frequency Feature ---
train_cc_counts = train.groupby('cc_num')[id_col].count().reset_index()
train_cc_counts.columns = ['cc_num', 'cc_num_txn_count']
train = train.merge(train_cc_counts, on='cc_num', how='left')

test_cc_counts = test.groupby('cc_num')[id_col].count().reset_index()
test_cc_counts.columns = ['cc_num', 'cc_num_txn_count']
test = test.merge(test_cc_counts, on='cc_num', how='left')

# --- Drop Unnecessary Columns ---
drop_cols = [id_col, 'trans_date', 'trans_time', 'unix_time', 'first', 'last', 'street', 'zip', 'lat', 'long', 
             'merch_lat', 'merch_long', 'dob', 'datetime', 'trans_num']
train.drop(columns=[c for c in drop_cols if c in train.columns], inplace=True)
test.drop(columns=[c for c in drop_cols if c in test.columns], inplace=True)

X = train.drop([target], axis=1)
y = train[target]
X_test = test.drop([target], axis=1, errors='ignore')

# --- Baseline Model Parameters ---
baseline_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# --- Cross-Validation ---
folds = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(
        baseline_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration)) / folds

# --- Evaluate Baseline ---
oof_auc = roc_auc_score(y, oof_preds)
print(f"Baseline OOF AUC: {oof_auc:.4f}")

# --- Save Submission ---
test_sub = pd.read_csv('test.csv', usecols=[id_col])
submission = sample_submission.copy()
submission[id_col] = test_sub[id_col]
submission['is_fraud'] = (test_preds > 0.5).astype(int)
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")