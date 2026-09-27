# ================================================================================
# SCRIPT BATCH INFERENZA SINCRONA (0-SHOT & FEW-SHOT) SU DEEPINFRA
# Benchmark Dataset: fanpage_test_1000.csv (1.000 articoli - 18 classi)
# Pool Few-Shot: fanpage_train_800.csv
# ================================================================================

import json
import os
import re
import sys
import time
import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    print('❌ Libreria openai non installata o datata. Esegui: pip install -q --upgrade openai')
    sys.exit(1)

# ==============================================================================
# 1. SETUP DEEPINFRA & CONFIGURAZIONE ESPERIMENTO
# ==============================================================================
try:
    from google.colab import userdata
    deepinfra_api_key = userdata.get('DEEPINFRA_API_KEY')
except Exception:
    deepinfra_api_key = os.environ.get('DEEPINFRA_API_KEY') or os.environ.get('DEEPINFRA_TOKEN')

if not deepinfra_api_key:
    deepinfra_api_key = input('⚠️ Inserisci la tua DEEPINFRA_API_KEY: ')

client = OpenAI(
    base_url='https://api.deepinfra.com/v1/openai',
    api_key=deepinfra_api_key,
)

# Protocollo: "0shot" oppure "fewshot"
EXECUTION_MODE = "0shot"     

# Tag specifico dell'esperimento per evitare sovrascritture o vecchi ripristini
EXPERIMENT_TAG = "news1000"  # I file si chiameranno: checkpoint-0shot-news1000-<model>.json

CHECKPOINT_EVERY = 20        # Frequenza salvataggio incrementale su disco

MODELS_TO_TEST = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-3-27b-it"
]

TEST_DATASET_PATH = 'fanpage_test_1000.csv'
TRAIN_DATASET_PATH = 'fanpage_train_800.csv'
CATEGORIES_FILE = 'main_categories.txt'
CHECKPOINT_DIR = 'fanpage_categorization_eval'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Tassonomia 18 classi
DEFAULT_CATEGORIES = [
    "attualita", "cinema", "cultura", "design", "diritto", "donna", 
    "economia", "esteri", "gossip", "innovazione", "music", "politica", 
    "scienze", "spettacolo", "sport", "stile-e-trend", "tech", "travel"
]

if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        CATEGORIES = [line.strip().lower() for line in f if line.strip()]
else:
    CATEGORIES = DEFAULT_CATEGORIES


# ==============================================================================
# 2. HELPER FUNZIONI PROMPT & NORMALIZZAZIONE
# ==============================================================================
def load_fewshot_exemplars(train_path: str, n_shots_per_class: int = 1) -> str:
    """Carica gli esempi per il few-shot dal dataset train_800."""
    if not os.path.exists(train_path):
        print(f"⚠️ Dataset train non trovato ({train_path}). Procedo in 0-shot.")
        return ""
    
    df_train = pd.read_csv(train_path)
    exemplars = []
    
    for cat in CATEGORIES:
        sample_rows = df_train[df_train['category'].str.strip().str.lower() == cat].head(n_shots_per_class)
        for _, row in sample_rows.iterrows():
            text_snippet = str(row['source']).strip()
            if len(text_snippet) > 400:
                text_snippet = text_snippet[:400] + "..."
            exemplars.append(f"Testo: \"{text_snippet}\"\nCategoria: {cat}")
            
    return "\n\n".join(exemplars)


def build_prompt(news_text: str, mode: str = "0shot", fewshot_context: str = "") -> str:
    """Costruisce il prompt in base al protocollo."""
    categories_str = ", ".join(CATEGORIES)
    
    if mode == "fewshot" and fewshot_context:
        return (
            "Classifica i testi informativi scegliendo UNA SOLA categoria tra quelle presenti in questa lista:\n"
            f"[{categories_str}]\n\n"
            "Di seguito sono forniti alcuni esempi di classificazione:\n\n"
            f"{fewshot_context}\n\n"
            "Ora classifica il seguente nuovo testo:\n"
            f'Testo: "{news_text}"\n\n'
            "Rispondi SOLTANTO con il nome della categoria scelta, senza punteggiatura o parole aggiuntive.\n"
            "Categoria:"
        )
    
    return (
        "Classifica il seguente testo informativo scegliendo UNA SOLA categoria "
        f"tra quelle presenti in questa lista: [{categories_str}].\n\n"
        f'Testo: "{news_text}"\n\n'
        "Rispondi SOLTANTO con il nome della categoria scelta, senza punteggiatura o parole aggiuntive.\n"
        "Categoria:"
    )


def predict_category(prompt: str, model_name: str) -> tuple[str, float]:
    """Chiamata API DeepInfra con retry automatizzato."""
    t0 = time.time()
    for attempt in range(1, 6):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=15
            )
            latency = time.time() - t0
            raw_response = response.choices[0].message.content.strip()
            return raw_response, latency

        except Exception as e:
            wait_time = 3.0 * (2 ** (attempt - 1))
            print(f"\n⚠️ [ERRORE API] (Tentativo {attempt}/5). Attendo {wait_time}s... Errore: {e}")
            time.sleep(wait_time)

    return "ERRORE", time.time() - t0


def clean_and_map_category(raw_pred: str) -> str:
    """Mapping e pulizia dell'etichetta predetta dall'LLM."""
    if not isinstance(raw_pred, str) or not raw_pred.strip():
        return "ALTRO"
    
    clean = raw_pred.lower().strip()
    clean = re.sub(r'^[^\w]+|[^\w]+$', '', clean)
    
    if clean in CATEGORIES:
        return clean
    
    for cat in CATEGORIES:
        if cat in clean:
            return cat
            
    return "ALTRO"


# ==============================================================================
# 3. PIPELINE DI ESECUZIONE SINCRONA
# ==============================================================================
def main():
    if not os.path.exists(TEST_DATASET_PATH):
        print(f"❌ File di test non trovato: {TEST_DATASET_PATH}.")
        sys.exit(1)

    df_test = pd.read_csv(TEST_DATASET_PATH)
    total_records = len(df_test)

    print(f"📊 Dataset Test Caricato: {total_records} articoli da '{TEST_DATASET_PATH}'.")
    print(f"🏷️ Tassonomia Categorie: {len(CATEGORIES)} classi")
    print(f"⚙️ Tag Esperimento: [{EXECUTION_MODE.upper()}-{EXPERIMENT_TAG.upper()}] | Checkpoint: ogni {CHECKPOINT_EVERY} record")

    fewshot_context = ""
    if EXECUTION_MODE == "fewshot":
        fewshot_context = load_fewshot_exemplars(TRAIN_DATASET_PATH, n_shots_per_class=1)
        print(f"💡 Contesto Few-Shot caricato da '{TRAIN_DATASET_PATH}'.")

    for selected_model in MODELS_TO_TEST:
        model_slug = selected_model.replace('/', '_').replace('-', '_').replace('.', '_')
        
        # Naming univoco del file di output per questo specifico esperimento sui 1000 record
        checkpoint_file = os.path.join(
            CHECKPOINT_DIR, 
            f'checkpoint-{EXECUTION_MODE}-{EXPERIMENT_TAG}-{model_slug}.json'
        )

        print("\n" + "=" * 80)
        print(f"🚀 AVVIO ESPERIMENTO [{EXECUTION_MODE.upper()}-{EXPERIMENT_TAG.upper()}] su [{selected_model}]")
        print(f"📁 Checkpoint destinazione: {checkpoint_file}")
        print("=" * 80)

        results = []
        processed_ids = set()

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    results = existing_data.get("results", [])
                    processed_ids = {str(r["id"]) for r in results}
                    print(f"🔄 Checkpoint ripristinato: {len(results)}/{total_records} completati.")
            except Exception as e:
                print(f"⚠️ Errore caricamento checkpoint ({e}). Si riparte da zero.")

        if len(results) >= total_records:
            print(f"✅ Esperimento per [{selected_model}] già COMPLETATO al 100% ({len(results)}/{total_records}). Salto al successivo.")
            continue

        fmt_data = {
            "metadata": {
                "model": {"provider": "deepinfra", "model_id": selected_model},
                "protocol": {"mode": EXECUTION_MODE, "experiment_tag": EXPERIMENT_TAG, "input_column": "source"},
                "dataset": {"total_records": total_records}
            },
            "results": results
        }

        # Filtro preventivo dei soli record ancora non processati
        unprocessed_df = df_test[~df_test['dataset_index'].astype(str).isin(processed_ids)]

        start_time = time.time()

        try:
            for idx, row in unprocessed_df.iterrows():
                if len(results) >= total_records:
                    break

                record_id = str(row.get("dataset_index", idx))
                news_text = str(row["source"])
                true_category = str(row["category"]).strip().lower()

                prompt = build_prompt(news_text, mode=EXECUTION_MODE, fewshot_context=fewshot_context)
                raw_prediction, latency = predict_category(prompt, selected_model)

                if raw_prediction == "ERRORE":
                    status = "error"
                    pred_category = "ERRORE"
                else:
                    status = "ok"
                    pred_category = clean_and_map_category(raw_prediction)

                record = {
                    "id": record_id,
                    "status": status,
                    "source_text": news_text,
                    "true_category": true_category,
                    "pred_category": pred_category,
                    "pred_raw": raw_prediction,
                    "latency_seconds": round(latency, 4)
                }

                results.append(record)
                fmt_data["results"] = results
                processed_ids.add(record_id)

                print(".", end="", flush=True)

                if len(results) % CHECKPOINT_EVERY == 0 or len(results) == total_records:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(fmt_data, f, ensure_ascii=False, indent=2)
                    print(f"\n💾 Checkpoint salvato: {len(results)}/{total_records} | File: {checkpoint_file}")

                time.sleep(0.1)

            # Salvataggio di chiusura
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(fmt_data, f, ensure_ascii=False, indent=2)

        except KeyboardInterrupt:
            print(f"\n🛑 Interrotto dall'utente durante [{selected_model}].")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(fmt_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Checkpoint salvato ({len(results)}/{total_records} salvati). Arresto del batch.")
            sys.exit(0)

        print(f"\n🎉 COMPLETATO [{EXECUTION_MODE.upper()}-{EXPERIMENT_TAG.upper()}] per [{selected_model}] in {time.time() - start_time:.1f}s")

    print("\n" + "=" * 80)
    print("🏁 TUTTI GLI ESPERIMENTI SONO STATI COMPLETATI CON SUCCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
