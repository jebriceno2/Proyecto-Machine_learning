import os, time, gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import warnings; warnings.filterwarnings('ignore')

# Limitar threads para evitar OOM
torch.set_num_threads(4)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEVICE     = torch.device('cpu')
MODEL_NAME = 'distilbert-base-multilingual-cased'
MAX_LEN    = 64
BATCH_SIZE = 32
CHUNK_SIZE = 2000   # guardar a disco cada 2000 muestras
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submissions')
os.makedirs(OUT_DIR, exist_ok=True)

print("Cargando datos...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
eval_df  = pd.read_csv(os.path.join(DATA_DIR, 'eval.csv'))

le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['decade'])
print(f"Train: {len(train_df)}  |  Eval: {len(eval_df)}  |  Clases: {len(le.classes_)}")

print(f"\nCargando {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
print("Modelo listo")


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1)


def extract_chunk(texts):
    enc = tokenizer(
        [str(t) for t in texts],
        max_length=MAX_LEN,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    with torch.no_grad():
        out = model(**enc)
    emb = mean_pool(out.last_hidden_state, enc['attention_mask'])
    return emb.numpy()


def extract_all(texts, desc):
    texts = list(texts)
    chunks = []
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        chunks.append(extract_chunk(batch))
        done = min(i + BATCH_SIZE, len(texts))
        if done % 1000 == 0 or done == len(texts):
            elapsed = time.time() - t0
            eta = elapsed / done * (len(texts) - done)
            print(f"  {desc}: {done}/{len(texts)}  elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min")
        gc.collect()
    return np.vstack(chunks)


# Rutas de cache
train_emb_path = os.path.join(OUT_DIR, 'X_train_distilbert.npy')
eval_emb_path  = os.path.join(OUT_DIR, 'X_eval_distilbert.npy')
y_path         = os.path.join(OUT_DIR, 'y_train.npy')

if os.path.exists(train_emb_path):
    print("\nCargando embeddings de entrenamiento (cache)...")
    X_train = np.load(train_emb_path)
else:
    print("\nExtrayendo embeddings de entrenamiento...")
    X_train = extract_all(train_df['text'], 'train')
    np.save(train_emb_path, X_train)
    np.save(y_path, train_df['label'].values)

if os.path.exists(eval_emb_path):
    print("Cargando embeddings de evaluacion (cache)...")
    X_eval = np.load(eval_emb_path)
else:
    print("Extrayendo embeddings de evaluacion...")
    X_eval = extract_all(eval_df['text'], 'eval')
    np.save(eval_emb_path, X_eval)

y_train = np.load(y_path)
print(f"\nX_train: {X_train.shape}  |  X_eval: {X_eval.shape}")

# Liberar modelo de memoria antes de entrenar SVC
del model
gc.collect()

print("\nEntrenando LinearSVC sobre embeddings...")
clf = LinearSVC(C=1.0, max_iter=3000, dual=True)
clf.fit(X_train, y_train)

preds  = clf.predict(X_eval)
decades = le.inverse_transform(preds)

submission = pd.DataFrame({'id': eval_df['id'], 'answer': decades})
sub_path   = os.path.join(OUT_DIR, 'submission_distilbert_svc.csv')
submission.to_csv(sub_path, index=False)
print(f"\nSubmission guardado: {sub_path}")
print(submission.head(10))
print(f"\nDistribucion:\n{submission['answer'].value_counts().sort_index()}")
