# #!/usr/bin/env python3
# """
# text_preproc_to_512_semantic.py

# Preprocess clinical text folders into 512-d semantic embeddings per patient
# and save outputs into:
#  C:\Users\haris\Downloads\project phase-1\training_dataset\semantic

# Usage (example):
#  python text_preproc_to_512_semantic.py \
#     --histories "C:\Users\haris\Downloads\project phase-1\dataset\TextData\TextData\histories_english" \
#     --reports  "C:\Users\haris\Downloads\project phase-1\dataset\TextData\TextData\reports_english" \
#     --surgery  "C:\Users\haris\Downloads\project phase-1\dataset\TextData\TextData\surgery_descriptions_english" \
#     --outdir "C:\Users\haris\Downloads\project phase-1\training_dataset\semantic" \
#     --use_transformer

# If --use_transformer is not given (or transformers/torch not installed),
# the script will use TF-IDF + TruncatedSVD fallback.
# """
import os, re, json, argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import h5py
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# optional transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TORCH = True
except Exception:
    TORCH = False

# ---------------- SETTINGS ----------------
DEFAULT_CLINICAL_H5 = Path(r"C:\Users\haris\Downloads\project phase-1\training_dataset\clinical_preprocessed_advanced.h5")
DEFAULT_OUTDIR = Path(r"C:\Users\haris\Downloads\project phase-1\training_dataset\semantic")
TRANSFORMER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
TARGET_DIM = 512
ENTITY_PATTERNS = {
    "tumor": [r"\btumou?r\b", r"\bcarcinoma\b", r"\bmalignan.*\b"],
    "metastasis": [r"\bmetastasi\w*\b", r"\bmetastatic\b"],
    "lymph_node": [r"\bnode\b", r"\blymp?h\b", r"\blymph\b"],
    "resection_margin_r0": [r"\bR0\b", r"\bcomplete resection\b", r"\bnegative margin\b"],
    "resection_margin_r1": [r"\bR1\b", r"\bpositive margin\b"],
    "tracheostomy": [r"\btracheostom\w*\b"],
    "peg": [r"\bPEG\b", r"\bgastrostomy\b"],
    "flap": [r"\bflap\b", r"\bgraft\b"],
    "complication": [r"\bcomplication\b", r"\binfection\b", r"\bfistula\b"]
}
NEGATION_TOKENS = {"no","not","without","denies","denied","negative","free of","no evidence","no sign"}

# ---------------- utils ----------------
def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(s):
    s = s.replace("\r\n", "\n")
    s = re.sub(r"<[^>]{1,80}>", " [REDACTED] ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sentence_split(text):
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_entities_and_negation(text, patterns=ENTITY_PATTERNS):
    text_l = text.lower()
    results = Counter()
    negated = Counter()
    for ent_name, regex_list in patterns.items():
        for patt in regex_list:
            for m in re.finditer(patt, text_l):
                span_start = m.start()
                window = text_l[max(0, span_start-60):span_start+1]
                is_neg = any(tok in window for tok in NEGATION_TOKENS)
                results[ent_name] += 1
                if is_neg:
                    negated[ent_name] += 1
    return dict(results), dict(negated)

def filename_to_patient_id(fname):
    base = Path(fname).stem
    m = re.search(r'(\d{1,5})$', base)
    if not m:
        m = re.search(r'_(\d{1,5})', base)
    if m:
        return m.group(1).zfill(3)
    return base

def try_load_clinical_ids(h5path):
    p = Path(h5path)
    if not p.exists(): 
        return None
    try:
        import h5py
        with h5py.File(str(p), "r") as f:
            raw = f["patient_id"][:]
            return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw]
    except Exception:
        return None

# Transformer helpers
def build_transformer(model_name=TRANSFORMER_MODEL, device="cpu"):
    if not TORCH:
        raise RuntimeError("Transformers/Torch not available")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def document_embedding_transformer(text, tokenizer, model, device="cpu", max_length=512):
    # mean pooling over token vectors (mask-aware)
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state.squeeze(0)   # (seq_len, hidden)
        amask = attention_mask.squeeze(0).unsqueeze(-1).float()
        sum_vec = (last_hidden * amask).sum(dim=0)
        denom = amask.sum(dim=0).clamp(min=1.0)
        emb = (sum_vec / denom).cpu().numpy()
    return emb

# ---------------- pipeline ----------------
def pipeline(histories_dir, reports_dir, surgery_dir, outdir, use_transformer=False, clinical_h5=None):
    histories = sorted(Path(histories_dir).glob("*.txt"))
    reports = sorted(Path(reports_dir).glob("*.txt"))
    surgery = sorted(Path(surgery_dir).glob("*.txt"))
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    clinical_ids = try_load_clinical_ids(clinical_h5) if clinical_h5 else None

    modalities = {
        "histories": histories,
        "reports": reports,
        "surgery": surgery
    }

    # read & map files -> pid -> list[text]
    docs = {m: defaultdict(list) for m in modalities.keys()}
    file_counts = {}
    for m, files in modalities.items():
        file_counts[m] = len(files)
        for f in files:
            txt = read_text_file(f)
            c = clean_text(txt)
            pid = filename_to_patient_id(f.name)
            if clinical_ids is not None and pid not in clinical_ids:
                digits = re.search(r'(\d{1,5})', f.name)
                if digits:
                    pid2 = digits.group(1).zfill(3)
                    if pid2 in clinical_ids:
                        pid = pid2
            docs[m][pid].append(c)

    # union of pids
    pids = sorted({p for m in docs.values() for p in m.keys()})
    n = len(pids)
    entity_names = sorted(list(ENTITY_PATTERNS.keys()))

    # load transformer if requested and available
    tokenizer = model = None
    device = "cpu"
    transformer_hidden = None
    if use_transformer and TORCH:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model = build_transformer(device=device)
        transformer_hidden = model.config.hidden_size

    # Prepare containers
    per_doc_embs = {m: defaultdict(list) for m in docs.keys()}
    per_doc_entity_counts = {m: defaultdict(list) for m in docs.keys()}
    per_doc_entity_neg = {m: defaultdict(list) for m in docs.keys()}
    per_doc_quals = {m: defaultdict(list) for m in docs.keys()}

    # If not using transformer, prepare TF-IDF + SVD per modality to later reduce to TARGET_DIM
    tfidf_vecs = {}
    svd_objs = {}
    if not use_transformer or not TORCH:
        for m in docs.keys():
            corpus = []
            for pid in pids:
                corpus.extend(docs[m].get(pid, []))
            vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
            if len(corpus) > 0:
                Xtf = vect.fit_transform(corpus)
                # SVD target dims: min(TARGET_DIM, Xtf.shape[1]-1)
                k = min(TARGET_DIM, max(2, Xtf.shape[1]-1))
                svd = TruncatedSVD(n_components=k, random_state=42)
                if Xtf.shape[1] > 1:
                    svd.fit(Xtf)
                tfidf_vecs[m] = vect
                svd_objs[m] = svd
            else:
                tfidf_vecs[m] = vect
                svd_objs[m] = None

    # Build per-document embeddings and features
    for m in docs.keys():
        for pid in pids:
            for doc in docs[m].get(pid, []):
                doc_len = len(doc.split())
                ent_counts, ent_neg = extract_entities_and_negation(doc)
                per_doc_entity_counts[m][pid].append(np.array([ent_counts.get(e,0) for e in entity_names], dtype=int))
                per_doc_entity_neg[m][pid].append(np.array([ent_neg.get(e,0) for e in entity_names], dtype=int))
                qual = np.log1p(doc_len) + sum(ent_counts.values())
                per_doc_quals[m][pid].append(float(qual))

                # embed
                if use_transformer and TORCH:
                    try:
                        emb = document_embedding_transformer(doc, tokenizer, model, device=device, max_length=512)
                    except Exception:
                        emb = np.zeros(transformer_hidden or TARGET_DIM)
                else:
                    tf = tfidf_vecs[m].transform([doc])
                    svd = svd_objs[m]
                    if svd is not None and tf.shape[1] > 0:
                        red = svd.transform(tf)
                        emb = red[0]
                    else:
                        emb = np.zeros(TARGET_DIM)
                per_doc_embs[m][pid].append(np.asarray(emb, dtype=float))

    # Per-modality: reduce document embeddings to TARGET_DIM via TruncatedSVD (fits on all doc embeddings for that modality)
    per_modality_proj = {}
    modality_doc_dim = {}
    for m in docs.keys():
        # collect all doc embeddings into matrix
        all_embs = []
        for pid in pids:
            all_embs.extend(per_doc_embs[m].get(pid, []))
        if len(all_embs) == 0:
            per_modality_proj[m] = None
            modality_doc_dim[m] = TARGET_DIM
            continue
        all_embs = np.stack(all_embs, axis=0)  # (n_docs, dim)
        orig_dim = all_embs.shape[1]
        modality_doc_dim[m] = orig_dim
        if orig_dim == TARGET_DIM:
            per_modality_proj[m] = ("identity", None)
        else:
            k = min(TARGET_DIM, orig_dim)
            svd = TruncatedSVD(n_components=k, random_state=42)
            svd.fit(all_embs)
            per_modality_proj[m] = ("svd", svd)

    # Now aggregate per-patient per-modality with clinical-topic attention pooling
    modality_patient_emb = {m: np.zeros((n, TARGET_DIM), dtype=np.float32) for m in docs.keys()}
    modality_patient_quality = {m: np.zeros((n,), dtype=np.float32) for m in docs.keys()}
    pid_to_idx = {pid:i for i,pid in enumerate(pids)}

    ALPHA = 1.0; BETA = 0.8
    for m in docs.keys():
        proj_kind, proj_obj = per_modality_proj[m]
        for pid in pids:
            idx = pid_to_idx[pid]
            doc_embs = per_doc_embs[m].get(pid, [])
            if len(doc_embs) == 0:
                modality_patient_quality[m][idx] = 0.0
                continue
            # project each doc emb to TARGET_DIM
            projected = []
            for emb in doc_embs:
                if proj_kind == "identity":
                    v = emb
                elif proj_kind == "svd":
                    v = proj_obj.transform(emb.reshape(1, -1))[0]
                    # if k < TARGET_DIM, pad zeros
                    if v.shape[0] < TARGET_DIM:
                        v = np.concatenate([v, np.zeros(TARGET_DIM - v.shape[0])], axis=0)
                else:
                    # fallback resize/truncate/pad
                    if emb.shape[0] >= TARGET_DIM:
                        v = emb[:TARGET_DIM]
                    else:
                        v = np.concatenate([emb, np.zeros(TARGET_DIM - emb.shape[0])], axis=0)
                projected.append(v)
            doc_emb_arr = np.stack(projected, axis=0)  # (D, TARGET_DIM)
            quals = np.array(per_doc_quals[m][pid], dtype=float)
            ent_arr = np.array(per_doc_entity_counts[m].get(pid, []), dtype=float)
            neg_arr = np.array(per_doc_entity_neg[m].get(pid, []), dtype=float)
            ent_density = ent_arr.sum(axis=1) if ent_arr.size else np.zeros(len(quals))
            neg_sum = neg_arr.sum(axis=1) if neg_arr.size else np.zeros(len(quals))
            score = np.log1p(quals + 1e-6) + ALPHA * ent_density - BETA * neg_sum
            if np.all(np.isfinite(score)):
                ex = np.exp(score - score.max())
                weights = ex / (ex.sum()+1e-8)
            else:
                weights = np.ones_like(score)/len(score)
            weighted = (weights[:,None] * doc_emb_arr).sum(axis=0)
            modality_patient_emb[m][idx] = weighted.astype(np.float32)
            modality_patient_quality[m][idx] = float(np.mean(quals))

    # Combined attention fusion across modalities using modality_quality as soft weights
    modality_list = ["histories","reports","surgery"]
    combined_emb = np.zeros((n, TARGET_DIM), dtype=np.float32)
    for i, pid in enumerate(pids):
        embs = np.stack([modality_patient_emb[m][i] for m in modality_list], axis=0)  # (3, dim)
        quals = np.array([modality_patient_quality[m][i] for m in modality_list], dtype=float)
        if np.all(embs==0):
            combined_emb[i] = np.zeros(TARGET_DIM, dtype=np.float32)
            continue
        wscore = np.log1p(quals + 1e-6)
        if np.all(np.isfinite(wscore)):
            ex = np.exp(wscore - np.nanmax(wscore))
            w = ex / (ex.sum()+1e-9)
        else:
            w = np.ones(len(quals))/len(quals)
        comb = (w[:,None] * embs).sum(axis=0)
        combined_emb[i] = comb.astype(np.float32)

    # Prepare entity_counts aggregated across modalities per patient
    entity_counts_total = np.zeros((n, len(entity_names)), dtype=np.int32)
    for m in docs.keys():
        for pid in pids:
            idx = pid_to_idx[pid]
            arrs = per_doc_entity_counts[m].get(pid, [])
            if len(arrs)>0:
                entity_counts_total[idx] += np.sum(np.stack(arrs, axis=0), axis=0).astype(np.int32)

    # file counts per modality
    file_counts_arr = np.zeros((n, len(modality_list)), dtype=np.int32)
    for j,m in enumerate(modality_list):
        for pid in pids:
            file_counts_arr[pid_to_idx[pid], j] = len(docs[m].get(pid, []))

    # Save HDF5
    out_h5 = Path(outdir) / "text_semantic_embeddings_512.h5"
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(out_h5), "w") as f:
        f.create_dataset("patient_id", data=np.array(pids, dtype="S"), dtype=dt)
        f.create_dataset("histories_embedding_512", data=modality_patient_emb["histories"])
        f.create_dataset("reports_embedding_512", data=modality_patient_emb["reports"])
        f.create_dataset("surgery_embedding_512", data=modality_patient_emb["surgery"])
        f.create_dataset("text_combined_embedding_512", data=combined_emb)
        f.create_dataset("modality_quality", data=np.stack([modality_patient_quality[m] for m in modality_list], axis=1).astype(np.float32))
        f.create_dataset("entity_names", data=np.array(entity_names, dtype="S"), dtype=dt)
        f.create_dataset("entity_counts", data=entity_counts_total)
        f.create_dataset("file_counts", data=file_counts_arr)
        f.create_dataset("modality_list", data=np.array(modality_list, dtype="S"), dtype=dt)
        f.attrs["n_patients"] = n
        f.attrs["target_dim"] = TARGET_DIM
        f.attrs["transformer_used"] = bool(use_transformer and TORCH)

    # Save preproc objects and summary
    obj = {
        "pids": pids,
        "per_modality_proj": {m: ("svd" if per_modality_proj[m] and per_modality_proj[m][0]=="svd" else "identity") for m in per_modality_proj},
        "transformer_model": TRANSFORMER_MODEL if (use_transformer and TORCH) else None,
        "projector_768_to_512": per_modality_proj["reports"][1] if per_modality_proj.get("reports") else None
    }
    joblib.dump(obj, Path(outdir) / "text_semantic_preproc_objects.joblib")

    summary = {
        "n_patients": n,
        "file_counts": {m: len(list(Path(d).glob('*.txt'))) for m,d in zip(modality_list, [histories_dir, reports_dir, surgery_dir])},
        "detected_modalities": modality_list,
        "entity_names": entity_names,
        "transformer_used": bool(use_transformer and TORCH),
        "output_h5": str(out_h5)
    }
    with open(Path(outdir) / "text_semantic_summary.json", "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    print("Saved: ", out_h5)
    print("Summary:", Path(outdir) / "text_semantic_summary.json")
    return

# --------------- CLi ---------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--histories", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--surgery", required=True)
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--use_transformer", action="store_true", help="Use Bio_ClinicalBERT (needs transformers & torch)")
    p.add_argument("--clinical_h5", default=str(DEFAULT_CLINICAL_H5), help="Optional clinical h5 to align patient ids")
    args = p.parse_args()
    pipeline(args.histories, args.reports, args.surgery, args.outdir, use_transformer=args.use_transformer, clinical_h5=args.clinical_h5)
