import os
import pathlib
from datetime import datetime
from dotenv import load_dotenv

from core.bench import bench
from core.dataset import DATASET_NAMES
from core.utils import load_config
from engines.openai_compatible import OpenAICompatibleEngine, OpenAICompatibleConfig

current_path = pathlib.Path(__file__).resolve().parent
config_root = current_path / "configs"
load_dotenv(current_path / ".env")
current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

def run_bench(tasks, limit, config):
    engine = OpenAICompatibleEngine(
        load_config(OpenAICompatibleConfig, config),
    )
    bench(
            engine=engine,
            tasks=tasks,
            limit=limit,
            save_outputs=True,
            close_engine=True,
            output_path=current_path / "outputs" / current_time,
    )
    
if __name__ == "__main__":
    start_time = datetime.now()
    limit = 5
    for provider_dir in config_root.iterdir():
        if provider_dir.is_dir():
            for config_file in provider_dir.glob("*.json"):
                print(f"Running benchmark for: {config_file}")
                try:
                    run_bench(
                        tasks=DATASET_NAMES,
                        limit=limit,
                        config=config_file
                    )
                except Exception as e:
                    print(f"Failed on {config_file}: {e}")

    end_time = datetime.now()
    delta_time = end_time - start_time
    print(f"Total time taken: {delta_time}")