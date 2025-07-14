import os
from core.bench import bench
from core.dataset import DATASET_NAMES
from core.utils import load_config
from engines.openai_compatible import OpenAICompatibleEngine, OpenAICompatibleConfig
from dotenv import load_dotenv

load_dotenv() #TODO use absolute path

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
    )
    
if __name__ == "__main__":
    run_bench(
        tasks = ["Github_easy"],
        limit = 5,
        config = "./configs/sambanova-llama-4-maverick.json"
    )
    