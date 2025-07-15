import os
from core.bench import bench
from core.dataset import DATASET_NAMES
from core.utils import load_config
from engines.openai_compatible import OpenAICompatibleEngine, OpenAICompatibleConfig
from dotenv import load_dotenv
from datetime import datetime
import pathlib

current_path = pathlib.Path(__file__).resolve().parent
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
    run_bench(
        tasks = DATASET_NAMES,
        limit = 5,
        config = current_path / "configs" / "sambanova-llama-4-maverick.json"
    )
    end_time = datetime.now()
    delta_time = end_time - start_time
    print(f"Ran in: {delta_time}")