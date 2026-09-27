# ================================================================================
# SCRIPT PER IL CALCOLO DELLE METRICHE DI CATEGORIZZAZIONE NEWS (18 CLASSI)
# ================================================================================

import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------------------------
# 1. PARAMETRI E SETUP TASSONOMIA
# ------------------------------------------------------------------------------
JSON_CHECKPOINT_PATH = 'fanpage_categorization_eval/checkpoint-0shot-news1000-meta_llama_Llama_3_3_70B_Instruct.json'
CATEGORIES_FILE = 'main_categories.txt'

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


# ------------------------------------------------------------------------------
# 2. CARICAMENTO DATI DAL CHECKPOINT JSON
# ------------------------------------------------------------------------------
def load_checkpoint_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"❌ File di checkpoint non trovato: {filepath}")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    results = data.get("results", [])

    print("\n" + "=" * 80)
    print(f"📄 RECORD CARICATI: {len(results)} da {filepath}")
    if metadata:
        model_id = metadata.get("model", {}).get("model_id", "N/A")
        protocol = metadata.get("protocol", {}).get("mode", "N/A")
        print(f"🤖 Modello: {model_id} | Protocollo: {protocol}")
    print("=" * 80)

    df = pd.DataFrame(results)
    return df, metadata


# ------------------------------------------------------------------------------
# 3. CALCOLO METRICHE & REPORT
# ------------------------------------------------------------------------------
def evaluate_predictions(df: pd.DataFrame):
    y_true = df['true_category'].str.strip().str.lower().values
    y_pred = df['pred_category'].str.strip().str.lower().values

    # Diagnostica anomalie / fuori tassonomia
    unmapped_count = (df['pred_category'].isin(['ALTRO', 'ERRORE'])).sum()
    coverage_rate = ((len(df) - unmapped_count) / len(df)) * 100
    avg_latency = df['latency_seconds'].mean() if 'latency_seconds' in df.columns else 0.0

    # Metriche Globali (Zero-division safe)
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    print("\n📊 METRICHE GLOBALI")
    print("-" * 50)
    print(f"• Accuracy:             {acc * 100:.2f}%")
    print(f"• Macro F1-Score:       {macro_f1 * 100:.2f}%")
    print(f"• Weighted F1-Score:    {w_f1 * 100:.2f}%")
    print(f"• Macro Precision:      {macro_p * 100:.2f}%")
    print(f"• Macro Recall:         {macro_r * 100:.2f}%")
    print(f"• Copertura Risposte:   {coverage_rate:.2f}% ({unmapped_count} unmapped/errori)")
    print(f"• Latenza Media:        {avg_latency:.3f} s/req")
    print("-" * 50)

    # Classification Report Dettagliato per Classe
    labels_to_show = [c for c in CATEGORIES if c in set(y_true) or c in set(y_pred)]
    if unmapped_count > 0 and 'altro' not in labels_to_show:
        labels_to_show.append('altro')

    report_str = classification_report(
        y_true, 
        y_pred, 
        labels=labels_to_show, 
        digits=4, 
        zero_division=0
    )
    
    print("\n📋 REPORT DETTAGLIATO PER CATEGORIA")
    print(report_str)

    # --------------------------------------------------------------------------
    # 4. MATRICE DI CONFUSIONE (PLOT & SAVE)
    # --------------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    cm_df = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)

    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matrice di Confusione - Classificazione News Fanpage', fontsize=14, pad=15)
    plt.xlabel('Categoria Predetta', fontsize=12)
    plt.ylabel('Categoria Reale (Ground Truth)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_img = JSON_CHECKPOINT_PATH.replace('.json', '_confusion_matrix.png')
    plt.savefig(output_img, dpi=300)
    print(f"\n🖼️ Matrice di Confusione salvata in: {output_img}")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        JSON_CHECKPOINT_PATH = sys.argv[1]

    df_results, meta = load_checkpoint_data(JSON_CHECKPOINT_PATH)
    evaluate_predictions(df_results)
