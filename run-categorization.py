# ================================================================================
# SCRIPT BATCH INFERENZA 0-SHOT NEWS CATEGORIZATION SU DEEPINFRA (LLAMA & GEMMA 3)
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
    print('❌ Libreria openai non installata. Esegui: pip install -q openai')
    sys.exit(1)

# ==============================================================================
# 1. SETUP DEEPINFRA & LISTA MODELLI DA TESTARE
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

# Lista modelli Llama e Gemma 3 da testare in sequenza
MODELS_TO_TEST = [
    # Modelli Llama
    
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    
    # Modelli Gemma 3
    "google/gemma-3-27b-it"
]

# Categorie consentite (frequenza > 15)
CATEGORIES = [
    "attualita", "politica", "sport", "diritto", "donna", "design",
    "music", "cultura", "cinema", "scienze", "tech", "sondaggi",
    "travel", "esteri", "stile-e-trend", "innovazione", "spettacolo"
]

TEST_DATASET_PATH = 'fanpage_categorized_found_only.csv'
CHECKPOINT_DIR = 'fanpage_categorization_eval'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================================================================
# 2. HELPER FUNZIONI PROMPT E NORMALIZZAZIONE RISPOSTA
# ==============================================================================
def predict_category(text: str, model_name: str) -> tuple[str, float]:
    """Genera il prompt 0-shot ed effettua la chiamata API a DeepInfra."""
    categories_str = ", ".join(CATEGORIES)
    
    prompt = (
        "Classifica il seguente testo informativo scegliendo UNA SOLA categoria "
        f"tra quelle presenti in questa lista: [{categories_str}].\n\n"
        f'Testo: "{text}"\n\n'
        "Rispondi SOLTANTO con il nome della categoria scelta, senza punteggiatura o parole aggiuntive.\n"
        "Categoria:"
    )

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
    """Pulisce la risposta raw dell'LLM e verifica se appartiene alle categorie valide."""
    if not isinstance(raw_pred, str) or not raw_pred.strip():
        return "ALTRO"
    
    # Normalizzazione minuscole e rimozione punteggiatura ai margini
    clean = raw_pred.lower().strip()
    clean = re.sub(r'^[^\w]+|[^\w]+$', '', clean)
    
    # Matching esatto
    if clean in CATEGORIES:
        return clean
    
    # Matching parziale se l'LLM risponde con una frase
    for cat in CATEGORIES:
        if cat in clean:
            return cat
            
    return "ALTRO"


# ==============================================================================
# 3. CARICAMENTO E FILTRAGGIO DATASET
# ==============================================================================
if not os.path.exists(TEST_DATASET_PATH):
    print(f"❌ File dataset non trovato: {TEST_DATASET_PATH}. Verificare il percorso.")
    sys.exit(1)

df_all = pd.read_csv(TEST_DATASET_PATH)

# Manteniamo solo i record con status == 'found' e url_category presente
df_test = df_all[
    (df_all['status'] == 'found') & 
    (df_all['url_category'].notna()) & 
    (df_all['url_category'].astype(str).str.strip() != '')
].copy().reset_index(drop=True)

total_records = len(df_test)
print(f'📊 Dataset Caricato e Filtrato: {total_records} notizie pronte per la classificazione 0-shot.')


# ==============================================================================
# 4. CICLO PRINCIPALE DI TEST MULTI-MODEL (0-SHOT)
# ==============================================================================
for selected_model in MODELS_TO_TEST:
    model_slug = selected_model.replace('/', '_').replace('-', '_').replace('.', '_')
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f'checkpoint-0shot-{model_slug}.json')

    print("\n" + "=" * 80)
    print(f"🚀 AVVIO ESPERIMENTO [0-SHOT] su [{selected_model}]")
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
                print(f"🔄 Checkpoint ripristinato: {len(results)}/{total_records} record già completati.")
        except Exception as e:
            print(f"⚠️ Errore caricamento checkpoint ({e}). Si riparte da zero per questo modello.")

    if len(results) == total_records:
        print(f"✅ Esperimento [0-SHOT] per [{selected_model}] già COMPLETATO al 100%. Salto al successivo.")
        continue

    fmt_data = {
        "metadata": {
            "model": {"provider": "deepinfra", "model_id": selected_model},
            "protocol": {"mode": "0shot", "input_column": "target"},
            "dataset": {"total_records": total_records}
        },
        "results": results
    }

    start_time = time.time()

    try:
        for idx, row in df_test.iterrows():
            # Utilizziamo l'indice o il dataset_index originale come ID
            record_id = str(row.get("dataset_index", idx))
            if record_id in processed_ids:
                continue

            # Testo estratto dalla colonna 'target'
            news_text = str(row["target"])
            true_category = str(row["url_category"]).strip().lower()

            raw_prediction, latency = predict_category(news_text, selected_model)
            
            if raw_prediction == "ERRORE":
                status = "error"
                pred_category = "ERRORE"
            else:
                status = "ok"
                pred_category = clean_and_map_category(raw_prediction)

            record = {
                "id": record_id,
                "status": status,
                "target_text": news_text,
                "true_category": true_category,
                "pred_category": pred_category,
                "pred_raw": raw_prediction,
                "latency_seconds": round(latency, 4)
            }

            results.append(record)
            fmt_data["results"] = results
            processed_ids.add(record_id)

            print(".", end="", flush=True)

            # Salvataggio incrementale ogni 20 record
            if len(results) % 20 == 0 or len(results) == total_records:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(fmt_data, f, ensure_ascii=False, indent=2)
                print(f"\n💾 Checkpoint salvato: {len(results)}/{total_records} | File: {checkpoint_file}")

            time.sleep(0.2)

    except KeyboardInterrupt:
        print(f"\n🛑 Interrotto dall'utente durante l'esecuzione su [{selected_model}].")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(fmt_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Checkpoint salvato ({len(results)}/{total_records} salvati). Arresto totale del batch.")
        sys.exit(0)

    print(f"\n🎉 COMPLETATO [0-SHOT] per [{selected_model}] in {time.time() - start_time:.1f}s")

print("\n" + "=" * 80)
print("🏁 TUTTI GLI ESPERIMENTI SU TUTTI I MODELLI SONO STATI COMPLETATI CON SUCCESSO!")
print("=" * 80)
