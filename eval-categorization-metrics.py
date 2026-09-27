# ==============================================================================
# SCRIPT BATCH METRICHE FORMATO FMT2 (0-SHOT / FEW-SHOT BENCHMARK)
# ==============================================================================

import glob
import json
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
CHECKPOINT_DIR = "fanpage_categorization_eval"
OUTPUT_SUMMARY_CSV = "metrics_summary_fmt2.csv"
CATEGORIES_FILE = "main_categories.txt"

# Tassonomia 18 classi
DEFAULT_CATEGORIES = [
    "attualita", "cinema", "cultura", "design", "diritto", "donna", 
    "economia", "esteri", "gossip", "innovazione", "music", "politica", 
    "scienze", "spettacolo", "sport", "stile-e-trend", "tech", "travel"
]

if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        CATEGORIES = sorted([line.strip().lower() for line in f if line.strip()])
else:
    CATEGORIES = sorted(DEFAULT_CATEGORIES)


# ==============================================================================
# ESTRAZIONE METRICHE DA CHECKPOINT FMT2
# ==============================================================================
def evaluate_checkpoint_fmt2(filepath: str) -> dict:
    """Legge un file JSON in formato fmt2 e calcola le metriche di classificazione."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    results = data.get("results", [])

    # Estrazione metadati fmt2
    model_id = metadata.get("model", {}).get("model_id", os.path.basename(filepath))
    provider = metadata.get("model", {}).get("provider", "N/A")
    protocol_mode = metadata.get("protocol", {}).get("mode", "N/A")
    experiment_tag = metadata.get("protocol", {}).get("experiment_tag", "N/A")
    total_records = metadata.get("dataset", {}).get("total_records", len(results))

    y_true = []
    y_pred = []
    latencies = []
    error_count = 0

    for r in results:
        if r.get("status") == "ok":
            y_true.append(str(r["true_category"]).strip().lower())
            y_pred.append(str(r["pred_category"]).strip().lower())
            if "latency_seconds" in r:
                latencies.append(r["latency_seconds"])
        else:
            error_count += 1

    if not y_true:
        return {
            "model_id": model_id,
            "provider": provider,
            "mode": protocol_mode,
            "tag": experiment_tag,
            "evaluated_samples": 0,
            "status": "No valid data"
        }

    # Calcolo Metriche Globale
    acc = accuracy_score(y_true, y_pred)
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    coverage_rate = (len(y_true) / total_records) * 100 if total_records > 0 else 0.0

    return {
        "model_id": model_id,
        "provider": provider,
        "mode": protocol_mode,
        "experiment_tag": experiment_tag,
        "total_expected": total_records,
        "evaluated_samples": len(y_true),
        "errors": error_count,
        "coverage_pct": round(coverage_rate, 2),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_macro": round(p_macro, 4),
        "recall_macro": round(r_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_weighted": round(p_weighted, 4),
        "recall_weighted": round(r_weighted, 4),
        "avg_latency_sec": round(avg_latency, 3),
        "file_name": os.path.basename(filepath)
    }


# ==============================================================================
# MAIN BATCH
# ==============================================================================
def main():
    # Cerca tutti i checkpoint JSON nella cartella di valutazione
    pattern = os.path.join(CHECKPOINT_DIR, "checkpoint-*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"⚠️ Nessun file trovato corrispondente al pattern '{pattern}'.")
        return

    print("=" * 100)
    print(f"📊 REPORT VALUTAZIONE METRICHE BATCH (FORMATO FMT2) | Tassonomia: {len(CATEGORIES)} classi")
    print("=" * 100)

    summary_list = []
    for file_path in sorted(json_files):
        try:
            metrics = evaluate_checkpoint_fmt2(file_path)
            summary_list.append(metrics)
        except Exception as e:
            print(f"⚠️ Errore durante l'elaborazione di {file_path}: {e}")

    df_metrics = pd.DataFrame(summary_list)
    
    if "f1_macro" in df_metrics.columns:
        df_metrics = df_metrics.sort_values(by=["mode", "f1_macro"], ascending=[True, False])

    # Salvataggio CSV
    df_metrics.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8")

    # Display tabellare
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n" + df_metrics.to_string(index=False))
    print("\n" + "=" * 100)
    print(f"💾 Report riepilogativo salvato in: {OUTPUT_SUMMARY_CSV}")
    print("=" * 100)


if __name__ == "__main__":
    main()
