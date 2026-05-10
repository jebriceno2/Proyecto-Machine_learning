import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submissions')

print("Cargando datos...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
eval_df  = pd.read_csv(os.path.join(DATA_DIR, 'eval.csv'))

le = LabelEncoder()
le.fit(train_df['decade'])

print("Cargando embeddings desde cache...")
X_train = np.load(os.path.join(OUT_DIR, 'X_train_distilbert.npy'))
X_eval  = np.load(os.path.join(OUT_DIR, 'X_eval_distilbert.npy'))
y_train = np.load(os.path.join(OUT_DIR, 'y_train.npy'))
print(f"X_train: {X_train.shape}  |  X_eval: {X_eval.shape}")

print("\nEntrenando LinearSVC (dual=False, C=0.5)...")
clf = LinearSVC(C=0.5, max_iter=1000, dual=False)
clf.fit(X_train, y_train)
print("Entrenamiento completado.")

preds   = clf.predict(X_eval)
decades = le.inverse_transform(preds)

submission = pd.DataFrame({'id': eval_df['id'], 'answer': decades})
sub_path   = os.path.join(OUT_DIR, 'submission_distilbert_svc.csv')
submission.to_csv(sub_path, index=False)
print(f"\nSubmission guardado: {sub_path}")
print(submission.head(10))
print(f"\nDistribución de predicciones:\n{submission['answer'].value_counts().sort_index()}")
