import os
import sys
import threading
from pathlib import Path
from tqdm import tqdm
from json import dumps
from dataclasses import asdict
from typing import List, Optional, Union

from core.engine import Engine
from core.evaluator import evaluate
from core.types import GenerationOutput
from core.dataset import Dataset, DatasetConfig
from core.utils import disable_print, nanoid, safe_min, print_scores, save_evaluation_results_to_csv, upload_to_s3
from core.messages import MessagesFormatter, FEW_SHOTS_MESSAGES_FORMATTER

write_lock = threading.Lock() 

def bench(
    engine: Engine,
    tasks: List[str],
    limit: Optional[int] = None,
    messages_formatter: Union[
        MessagesFormatter, List[MessagesFormatter]
    ] = FEW_SHOTS_MESSAGES_FORMATTER,
    close_engine: bool = True,
    save_outputs: bool = False,
    output_path: Optional[str] = "outputs"
) -> List[List[GenerationOutput]]:
    """Benchmarks an engine with specified tasks and datasets.

    :param engine: Engine
        The engine to benchmark.
    :param tasks: List[str]
        The tasks to benchmark.
    :param limit: Optional[int]
        The limit on the number of samples to benchmark.
    :param messages_formatter: Union[MessagesFormatter, List[MessagesFormatter]]
        The function(s) to format the schema into a list of messages. If a single
        function is provided, it will be used for all tasks. If a list of
        functions is provided, each function will be used for the corresponding
        task.
    :param close_engine: bool
        Whether to close the engine after the benchmark.
    :param save_outputs: bool
        Whether to save the generation outputs after the benchmark.
    :param output_path: Optional[str]
        where to save the generation results

    :return: List[List[GenerationOutput]]
        The generation outputs for each sample for each task.
    """
    id = nanoid()
    
    if save_outputs:
        output_path = Path(output_path)
        engine_dir = output_path / engine.name
        engine_dir.mkdir(parents=True, exist_ok=True)

        if engine.name == "openai_compatible":
            provider_dir = engine_dir / engine.config.provider
            provider_dir.mkdir(parents=True, exist_ok=True)
            save_json_output_path = provider_dir / f"{engine.config.tokenizer.replace('/', '_')}.jsonl"
        else:
            save_json_output_path = engine_dir / f"{id}.jsonl"

        with open(save_json_output_path, "w") as f:
            f.write(f"{dumps({'engine': engine.name, 'engine_config': asdict(engine.config)})}\n")
    

    if not isinstance(messages_formatter, list):
        messages_formatter = [messages_formatter] * len(tasks)

    all_outputs = []
    compliance = []
    perf_metrics = []
    output_tokens = []
    declared_coverage = []
    empirical_coverage = []
    
    for task, mf in zip(tasks, messages_formatter):
        print(f"# Running Task {task} for engine {engine.name}, with provider {getattr(engine.config, 'provider', 'n/a')} for model {getattr(engine.config, 'model', 'n/a')}")
        task_outputs = []
        dataset = Dataset(DatasetConfig(task, limit=limit))
        for messages, schema in tqdm(
            dataset.iter(mf),
            total=safe_min(len(dataset), limit),
            desc=task,
            file=sys.stdout,
        ):
            with disable_print(): # comment to debug
                schema = engine.adapt_schema(schema)
                result = engine.generate(task, messages, schema)
                task_outputs.append(result)
        
        dc, ec, cl, pm, ot, evaluated_outputs = evaluate(task_outputs)
        declared_coverage.append(dc)
        empirical_coverage.append(ec)
        compliance.append(cl)
        perf_metrics.append(pm)
        output_tokens.append(ot)
        
        if save_outputs:
            with open(save_json_output_path, "a") as f:
                for output in evaluated_outputs:
                    f.write(f"{dumps(asdict(output))}\n")
                    
            results_path = engine_dir / "eval_results.csv"
            save_evaluation_results_to_csv(
                csv_path=results_path,
                run_id=output_path.name or id,
                provider = getattr(engine.config, "provider", "n/a"),
                model = getattr(engine.config, "tokenizer", "n/a"),
                task=task,
                dc=dc,
                ec=ec,
                cl=cl,
                pm=pm,  
                ot=ot,
                write_lock=write_lock
            )
            s3_path=f"fc-so-testing-suite/jsonschemabench_snova/{'/'.join(results_path.parts[-3:])}"
            upload_to_s3(results_path, s3_path)
            
        all_outputs.append(evaluated_outputs)
        
    print_scores(
        declared_coverage,
        empirical_coverage,
        compliance,
        perf_metrics,
        output_tokens,
        tasks,
    )

    if close_engine:
        engine.close()

    return all_outputs
