import argparse
import json
import pathlib
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_env import resolve_config_env, write_sidecar
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

def run_bench(tasks, limit, config_path, output_path, config_env=None):
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
            config_env=config_env,
        )
    except Exception as e:
        print(f"[BENCH] Error running benchmark for config {config_path.name}: {e}")

def collect_base_urls(config_file):
    """Record the endpoint actually used per provider, for the sidecar audit trail.

    config_env is a label set by hand, so recording the real base_url is what makes a
    mislabelled run provable after the fact.
    """
    base_urls = {}
    for provider, models in config_file.items():
        for model in models:
            try:
                cfg = json.loads((CONFIG_ROOT / model).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            url = cfg.get("base_url")
            if url:
                base_urls.setdefault(provider, url)
                break
    return base_urls

def run_provider_benchmarks(provider, models, output_path, config_env=None):
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
            config_env=config_env,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Run JSONSchemaBench across providers.")
    parser.add_argument(
        "--config-env",
        type=str,
        default=None,
        help="Config env id for this run (overrides config_env in configs/config.json)",
    )
    return parser.parse_args()

def main(cli_config_env=None):
    load_dotenv(CURRENT_PATH / ".env")
    config_file = load_json_config(CONFIG_FILE_PATH)

    # Non-provider keys must be removed before iterating: every remaining key is
    # treated as a provider name.
    config_env = resolve_config_env(cli_config_env, config_file)
    config_file.pop("config_env", None)
    config_file.pop("_comment", None)
    print(f"[CONFIG] config_env: {config_env}")

    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = OUTPUT_DIR / current_time

    # Sidecar in the run dir; the report script locates runs by this same id.
    write_sidecar(output_path, config_env, collect_base_urls(config_file))

    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                run_provider_benchmarks, provider, models, output_path, config_env
            ): provider
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
    main(cli_config_env=parse_args().config_env)
