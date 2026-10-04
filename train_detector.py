"""
Trains the fake audio detector used by the web app on the fair dataset (data/fair,
built by prepare_fair_test.py) and saves it with its evaluation metrics:
- WebApp/models/detector_fair.keras: model trained on all 300 speakers
- WebApp/models/detector_metrics.json: 5-fold speaker-grouped cross-validation scores

Usage:
    python train_detector.py
"""
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import functions

OUT_DIR = Path("WebApp/models")


def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(193,)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def fit(X, y, groups, idx, seed):
    """Trains on idx, holding out 10% of its speakers for early stopping."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    a, b = next(gss.split(idx, groups=groups[idx]))
    model = build_model()
    model.fit(X[idx[a]], y[idx[a]], epochs=1000, verbose=0,
              validation_data=(X[idx[b]], y[idx[b]]),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                          restore_best_weights=True)])
    return model


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    files, labels = [], []
    for folder, label in [('data/fair/cloned_audio', 0), ('data/fair/real_audio', 1)]:
        for f in sorted(os.listdir(folder)):
            files.append(f'{folder}/{f}')
            labels.append(label)
    y = np.array(labels)
    groups = np.array([Path(f).stem for f in files])

    print(f"Extracting features from {len(files)} files...")
    X = functions.concat_features(functions.extract_features(files))

    y_true, y_pred, fold_f1 = [], [], []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        model = fit(X, y, groups, train_idx, seed=fold)
        pred = np.rint(model.predict(X[test_idx], verbose=0)).flatten().astype(int)
        fold_f1.append(f1_score(y[test_idx], pred))
        y_true.extend(y[test_idx])
        y_pred.extend(pred)
        print(f"Fold {fold + 1}: F1 {fold_f1[-1]:.3f}")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'dataset': '300 TIMIT speakers: one real recording and one SV2TTS clone each',
        'evaluation': '5-fold cross-validation grouped by speaker',
        'positive_class': 'Real',
        'f1': float(np.mean(fold_f1)),
        'f1_std': float(np.std(fold_f1, ddof=1)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'confusion': {'real_as_real': int(tp), 'real_as_clone': int(fn),
                      'clone_as_clone': int(tn), 'clone_as_real': int(fp)},
        'n_files': len(files),
    }
    print(json.dumps(metrics, indent=2))

    model = fit(X, y, groups, np.arange(len(y)), seed=42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUT_DIR / 'detector_fair.keras')
    (OUT_DIR / 'detector_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()
