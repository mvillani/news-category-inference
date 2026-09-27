# ================================================================================
# SCRIPT BATCH INFERENZA ASINCRONA (0-SHOT & FEW-SHOT) SU DEEPINFRA
# Benchmark Dataset: fanpage_test_1000.csv (1.000 articoli - 18 classi)
# Pool Few-Shot: fanpage_train_800.csv
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

# Parametri di Concorrenza e Batch
MAX_CONCURRENT_REQUESTS = 10  # Numero di chiamate HTTP simultanee
CHUNK_SIZE = 20              # Dimensione del blocco prima di salvare il checkpoint

# Modalità di esperimento da eseguire: "0shot" oppure "fewshot"
EXECUTION_MODE = "0shot"     # Cambiare in "fewshot" per il protocollo Few-Shot

# Modelli da testare su DeepInfra
MODELS_TO_TEST = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-3-27b-it"
]

# Percorsi File
TEST_DATASET_PATH = 'fanpage_test_1000.csv'
TRAIN_DATASET_PATH = 'fanpage_train_800.csv'
CATEGORIES_FILE = 'main_categories.txt'
CHECKPOINT_DIR = 'fanpage_categorization_eval'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Categorie predefinite (18 classi dal documento)
DEFAULT_CATEGORIES = [
    "attualita", "cinema", "cultura", "design", "diritto", "donna", 
    "economia", "esteri", "gossip", "innovazione", "music", "politica", 
    "scienze", "spettacolo", "sport", "stile-e-trend", "tech", "travel"
]

# Caricamento dinamico se main_categories.txt esiste
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        CATEGORIES = [line.strip().lower() for line in f if line.strip()]
else:
    CATEGORIES = DEFAULT_CATEGORIES


# ==============================================================================
# 2. HELPER PROMPT & FEW-SHOT EXEMPLARS GENERATOR
# ==============================================================================
def load_fewshot_exemplars(train_path: str, n_shots_per_class: int = 1) -> str:
    """Preleva fino a n_shots_per_class dal dataset di train da usare come esempi in-context."""
    if not os.path.exists(train_path):
        print(f"⚠️ Dataset train non trovato ({train_path}). Procedo senza esempi.")
        return ""
    
    df_train = pd.read_csv(train_path)
    exemplars = []
    
    for cat in CATEGORIES:
        sample_rows = df_train[df_train['category'].str.strip().str.lower() == cat].head(n_shots_per_class)
        for _, row in sample_rows.iterrows():
            text_snippet = str(row['source']).strip()
            # Troncamento preventivo se l'articolo di train è troppo lungo
            if len(text_snippet) > 400:
                text_snippet = text_snippet[:400] + "..."
            exemplars.append(f"Testo: \"{text_snippet}\"\nCategoria: {cat}")
            
    return "\n\n".join(exemplars)


def build_prompt(news_text: str, mode: str = "0shot", fewshot_context: str = "") -> str:
    """Costruisce il prompt in base al protocollo (0-shot o few-shot)."""
    categories_str = ", ".join(CATEGORIES)
    
    if mode == "fewshot" and fewshot_context:
        prompt = (
            "Classifica i testi informativi scegliendo UNA SOLA categoria tra quelle presenti in questa lista:\n"
            f"[{categories_str}]\n\n"
            "Di seguito sono forniti alcuni esempi di classificazione:\n\n"
            f"{fewshot_context}\n\n"
            "Ora classifica il seguente nuovo testo:\n"
            f'Testo: "{news_text}"\n\n'
            "Rispondi SOLTANTO con il nome della categoria scelta, senza punteggiatura o parole aggiuntive.\n"
            "Categoria:"
        )
    else:  # Default 0-shot
        prompt = (
            "Classifica il seguente testo informativo scegliendo UNA SOLA categoria "
            f"tra quelle presenti in questa lista: [{categories_str}].\n\n"
            f'Testo: "{news_text}"\n\n'
            "Rispondi SOLTANTO con il nome della categoria scelta, senza punteggiatura o parole aggiuntive.\n"
            "Categoria:"
        )
        
    return prompt


def clean_and_map_category(raw_pred: str) -> str:
    """Normalizza e valida l'output grezzo dell'LLM."""
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
# 3. WORKER ASINCRONO PER SINGOLO RECORD
# ==============================================================================
async def predict_category_async(
    prompt: str, 
    model_name: str, 
    semaphore: asyncio.Semaphore
) -> tuple[str, float]:
    """Effettua la chiamata API asincrona gestendo i retry con exponential backoff."""
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


async def process_record(
    row: pd.Series, 
    idx: int, 
    selected_model: str, 
    mode: str, 
    fewshot_context: str, 
    semaphore: asyncio.Semaphore
) -> dict:
    """Prepara il record, esegue l'inferenza e mappa il risultato."""
    record_id = str(row.get("dataset_index", idx))
    news_text = str(row["source"])
    true_category = str(row["category"]).strip().lower()

    prompt = build_prompt(news_text, mode=mode, fewshot_context=fewshot_context)
    raw_prediction, latency = await predict_category_async(prompt, selected_model, semaphore)

    if raw_prediction == "ERRORE":
        status = "error"
        pred_category = "ERRORE"
    else:
        status = "ok"
        pred_category = clean_and_map_category(raw_prediction)

    return {
        "id": record_id,
        "status": status,
        "source_text": news_text,
        "true_category": true_category,
        "pred_category": pred_category,
        "pred_raw": raw_prediction,
        "latency_seconds": round(latency, 4)
    }


# ==============================================================================
# 4. MAIN ASINCRONO
# ==============================================================================
async def main_async():
    if not os.path.exists(TEST_DATASET_PATH):
        print(f"❌ File di test non trovato: {TEST_DATASET_PATH}.")
        sys.exit(1)

    df_test = pd.read_csv(TEST_DATASET_PATH)
    total_records = len(df_test)

    print(f"📊 Dataset Test Caricato: {total_records} articoli dal file '{TEST_DATASET_PATH}'.")
    print(f"🏷️ Numero Categorie Tassonomia: {len(CATEGORIES)}")
    print(f"⚙️ Protocollo: [{EXECUTION_MODE.upper()}] | Concorrenza max: {MAX_CONCURRENT_REQUESTS} | Checkpoint batch: {CHUNK_SIZE}")

    # Carica esempi few-shot dal train set se abilitato
    fewshot_context = ""
    if EXECUTION_MODE == "fewshot":
        fewshot_context = load_fewshot_exemplars(TRAIN_DATASET_PATH, n_shots_per_class=1)
        print(f"💡 Contesto Few-Shot caricato con successo dal pool '{TRAIN_DATASET_PATH}'.")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    for selected_model in MODELS_TO_TEST:
        model_slug = selected_model.replace('/', '_').replace('-', '_').replace('.', '_')
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f'checkpoint-{EXECUTION_MODE}-{model_slug}.json')

        print("\n" + "=" * 80)
        print(f"🚀 AVVIO ESPERIMENTO [{EXECUTION_MODE.upper()}] su [{selected_model}]")
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
                    print(f"🔄 Checkpoint ripristinato: {len(results)}/{total_records} record completati.")
            except Exception as e:
                print(f"⚠️ Errore caricamento checkpoint ({e}). Si riparte da zero.")

        if len(results) == total_records:
            print(f"✅ Esperimento per [{selected_model}] già COMPLETATO al 100%. Salto al successivo.")
            continue

        fmt_data = {
            "metadata": {
                "model": {"provider": "deepinfra", "model_id": selected_model},
                "protocol": {"mode": EXECUTION_MODE, "input_column": "source"},
                "dataset": {"total_records": total_records}
            },
            "results": results
        }

        # Selezione dei record non ancora processati
        unprocessed_rows = []
        for idx, row in df_test.iterrows():
            record_id = str(row.get("dataset_index", idx))
            if record_id not in processed_ids:
                unprocessed_rows.append((idx, row))

        start_time = time.time()

        # Esecuzione per lotti (chunk)
        for i in range(0, len(unprocessed_rows), CHUNK_SIZE):
            chunk = unprocessed_rows[i:i + CHUNK_SIZE]
            
            tasks = [
                process_record(row, idx, selected_model, EXECUTION_MODE, fewshot_context, semaphore) 
                for idx, row in chunk
            ]

            try:
                chunk_results = await asyncio.gather(*tasks)
                results.extend(chunk_results)
                fmt_data["results"] = results
                print(".", end="", flush=True)

                # Salvataggio sincrono incrementale
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(fmt_data, f, ensure_ascii=False, indent=2)

                print(f"\n💾 Checkpoint salvato: {len(results)}/{total_records} | File: {checkpoint_file}")

            except KeyboardInterrupt:
                print(f"\n🛑 Interrotto dall'utente durante [{selected_model}].")
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(fmt_data, f, ensure_ascii=False, indent=2)
                print(f"💾 Checkpoint salvato ({len(results)}/{total_records} salvati). Arresto batch.")
                sys.exit(0)

        print(f"\n🎉 COMPLETATO [{EXECUTION_MODE.upper()}] per [{selected_model}] in {time.time() - start_time:.1f}s")

    print("\n" + "=" * 80)
    print("🏁 TUTTI GLI ESPERIMENTI SU TUTTI I MODELLI SONO STATI COMPLETATI CON SUCCESSO!")
    print("=" * 80)

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n🛑 Programma interrotto dall'utente.")
