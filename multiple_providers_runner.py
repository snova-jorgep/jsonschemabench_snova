import json
import pathlib
from datetime import datetime
from dotenv import load_dotenv

from core.bench import bench
from core.dataset import DATASET_NAMES
from core.utils import load_config
from engines.openai_compatible import OpenAICompatibleEngine, OpenAICompatibleConfig


CURRENT_PATH = pathlib.Path(__file__).resolve().parent
CONFIG_ROOT = CURRENT_PATH / "configs"
CONFIG_FILE_PATH = CONFIG_ROOT / "config.json"
OUTPUT_DIR = CURRENT_PATH / "outputs"
LIMIT = 1

def load_json_config(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            return config_data
    except FileNotFoundError:
        print(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
    return {}

def run_bench(tasks, limit, config_path, output_path):
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

def main():
    load_dotenv(CURRENT_PATH / ".env")
    config_file = load_json_config(CONFIG_FILE_PATH)
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = OUTPUT_DIR / current_time

    start_time = datetime.now()
    
    for provider, models in config_file.items():
        for model in models:
            print(f"Running benchmark for: {model}")
            try:
                run_bench(
                    tasks=DATASET_NAMES,
                    limit=LIMIT,
                    config_path=CONFIG_ROOT / model,
                    output_path=output_path
                )
            except Exception as e:
                print(f"Benchmark failed for {model}: {e}")
    
    delta_time = datetime.now() - start_time
    print(f"Total time taken: {delta_time}")

if __name__ == "__main__":
    main()
