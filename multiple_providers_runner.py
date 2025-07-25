import json
import pathlib
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.bench import bench
from core.dataset import DATASET_NAMES
from core.utils import load_config
from engines.openai_compatible import OpenAICompatibleEngine, OpenAICompatibleConfig

CURRENT_PATH = pathlib.Path(__file__).resolve().parent
CONFIG_ROOT = CURRENT_PATH / "configs"
CONFIG_FILE_PATH = CONFIG_ROOT / "config.json"
OUTPUT_DIR = CURRENT_PATH / "outputs"
LIMIT = 100

def load_json_config(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[CONFIG] Error loading config: {e}")
    return {}

def run_bench(tasks, limit, config_path, output_path):
    try:
        config = load_config(OpenAICompatibleConfig, config_path)
        engine = OpenAICompatibleEngine(config)
        bench(
            engine=engine,
            tasks=tasks,
            limit=limit,
            save_outputs=True,
            close_engine=True,
            output_path=output_path,
        )
    except Exception as e:
        print(f"[BENCH] Error running benchmark for config {config_path.name}: {e}")

def run_provider_benchmarks(provider, models, output_path):
    provider_output_path = output_path
    provider_output_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        print(f"[{provider}] Running benchmark for model: {model}")
        datasets_to_run = DATASET_NAMES.copy()
        datasets_to_run.remove("Github_ultra")
        run_bench(
            tasks=datasets_to_run,
            limit=LIMIT,
            config_path=CONFIG_ROOT / model,
            output_path=provider_output_path,
        )

def main():
    load_dotenv(CURRENT_PATH / ".env")
    config_file = load_json_config(CONFIG_FILE_PATH)
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = OUTPUT_DIR / current_time

    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_provider_benchmarks, provider, models, output_path): provider
            for provider, models in config_file.items()
        }
        for future in as_completed(futures):
            provider = futures[future]
            try:
                future.result()
                print(f"[{provider}] Benchmarks completed successfully.")
            except Exception as e:
                print(f"[{provider}] Unexpected failure in provider-level task: {e}")

    delta_time = datetime.now() - start_time
    print(f"Total time taken: {delta_time}")

if __name__ == "__main__":
    main()
