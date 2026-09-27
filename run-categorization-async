# ================================================================================
# SCRIPT BATCH INFERENZA ASINCRONA (0-SHOT) DEEPINFRA (LLAMA & GEMMA 3)
# ================================================================================

import asyncio
import json
import os
import re
import sys
import time
import pandas as pd

try:
    from openai import AsyncOpenAI
except ImportError:
    print('❌ Libreria openai non installata o datata. Esegui: pip install -q --upgrade openai')
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

# Client Asincrono OpenAI per DeepInfra
async_client = AsyncOpenAI(
    base_url='https://api.deepinfra.com/v1/openai',
    api_key=deepinfra_api_key,
)

# Parametri di Concorrenza
MAX_CONCURRENT_REQUESTS = 10  # Numero di chiamate HTTP simultanee (Pool/Semaphore)
CHUNK_SIZE = 20              # Dimensione del blocco di elaborazione prima di salvare il checkpoint

MODELS_TO_TEST = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-3-27b-it"
]

CATEGORIES = [
    "attualita", "politica", "sport", "diritto", "donna", "design",
    "music", "cultura", "cinema", "scienze", "tech", "sondaggi",
    "travel", "esteri", "stile-e-trend", "innovazione", "spettacolo"
]

TEST_DATASET_PATH = 'fanpage_categorized_found_only.csv'
CHECKPOINT_DIR = 'fanpage_categorization_eval'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================================================================
# 2. HELPER FUNZIONI PROMPT, CLEANING & WORKER ASINCRONO
# ==============================================================================
def clean_and_map_category(raw_pred: str) -> str:
    """Pulisce la risposta raw dell'LLM e verifica se appartiene alle categorie valide."""
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


async def predict_category_async(text: str, model_name: str, semaphore: asyncio.Semaphore) -> tuple[str, float]:
    """Genera la chiamata API asincrona gestendo i tentativi di retry e il semaforo."""
    categories_str = ", ".join(CATEGORIES)
    prompt = (
        "Classifica il seguente testo informativo scegliendo UNA SOLA categoria "
        f"tra quelle presenti in questa lista: [{categories_str}].\n\n"
        f'Testo: "{text}"\n\n'
        "Rispondi SOLTANTO con il nome della categoria scelta, senza punteggiatura o parole aggiuntive.\n"
        "Categoria:"
    )

    async with semaphore:
        t0 = time.time()
        for attempt in range(1, 6):
            try:
                response = await async_client.chat.completions.create(
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
                await asyncio.sleep(wait_time)

        return "ERRORE", time.time() - t0


async def process_record(row: pd.Series, idx: int, selected_model: str, semaphore: asyncio.Semaphore) -> dict:
    """Processa il singolo record estraendo i campi e formattando il dizionario di output."""
    record_id = str(row.get("dataset_index", idx))
    news_text = str(row["target"])
    true_category = str(row["url_category"]).strip().lower()

    raw_prediction, latency = await predict_category_async(news_text, selected_model, semaphore)

    if raw_prediction == "ERRORE":
        status = "error"
        pred_category = "ERRORE"
    else:
        status = "ok"
        pred_category = clean_and_map_category(raw_prediction)

    return {
        "id": record_id,
        "status": status,
        "target_text": news_text,
        "true_category": true_category,
        "pred_category": pred_category,
        "pred_raw": raw_prediction,
        "latency_seconds": round(latency, 4)
    }

# ==============================================================================
# 3. PIPELINE ASINCRONA PRINCIPALE
# ==============================================================================
async def main_async():
    if not os.path.exists(TEST_DATASET_PATH):
        print(f"❌ File dataset non trovato: {TEST_DATASET_PATH}. Verificare il percorso.")
        sys.exit(1)

    df_all = pd.read_csv(TEST_DATASET_PATH)

    df_test = df_all[
        (df_all['status'] == 'found') & 
        (df_all['url_category'].notna()) & 
        (df_all['url_category'].astype(str).str.strip() != '')
    ].copy().reset_index(drop=True)

    total_records = len(df_test)
    print(f'📊 Dataset Caricato e Filtrato: {total_records} notizie pronte per la classificazione.')
    print(f'⚙️ Concorrenza massima: {MAX_CONCURRENT_REQUESTS} task | Batch checkpoint: ogni {CHUNK_SIZE} record')

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    for selected_model in MODELS_TO_TEST:
        model_slug = selected_model.replace('/', '_').replace('-', '_').replace('.', '_')
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f'checkpoint-0shot-{model_slug}.json')

        print("\n" + "=" * 80)
        print(f"🚀 AVVIO ESPERIMENTO [0-SHOT ASINCRONO] su [{selected_model}]")
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
            print(f"✅ Esperimento per [{selected_model}] già COMPLETATO al 100%. Salto al successivo.")
            continue

        fmt_data = {
            "metadata": {
                "model": {"provider": "deepinfra", "model_id": selected_model},
                "protocol": {"mode": "0shot", "input_column": "target"},
                "dataset": {"total_records": total_records}
            },
            "results": results
        }

        # Selezione dei soli record non ancora processati
        unprocessed_rows = []
        for idx, row in df_test.iterrows():
            record_id = str(row.get("dataset_index", idx))
            if record_id not in processed_ids:
                unprocessed_rows.append((idx, row))

        start_time = time.time()

        # Suddivisione dei record rimanenti in lotti (chunk)
        for i in range(0, len(unprocessed_rows), CHUNK_SIZE):
            chunk = unprocessed_rows[i:i + CHUNK_SIZE]
            
            # Creazione delle coroutine per il lotto corrente
            tasks = [
                process_record(row, idx, selected_model, semaphore) 
                for idx, row in chunk
            ]

            try:
                # Esecuzione in parallelo del lotto corrente
                chunk_results = await asyncio.gather(*tasks)
                
                results.extend(chunk_results)
                fmt_data["results"] = results
                print(".", end="", flush=True)

                # Salvataggio incrementale sincrono alla fine del lotto
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(fmt_data, f, ensure_ascii=False, indent=2)

                print(f"\n💾 Checkpoint salvato: {len(results)}/{total_records} | File: {checkpoint_file}")

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

# ==============================================================================
# 4. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n🛑 Programma interrotto dall'utente.")
