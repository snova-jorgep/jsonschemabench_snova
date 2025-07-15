import os
import sys
from tqdm import tqdm
from json import dumps
from dataclasses import asdict
from typing import List, Optional, Union

from core.engine import Engine
from core.evaluator import evaluate
from core.types import GenerationOutput
from core.dataset import Dataset, DatasetConfig
from core.utils import disable_print, nanoid, safe_min, print_scores
from core.messages import MessagesFormatter, FEW_SHOTS_MESSAGES_FORMATTER


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
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        engine_dir = f"{output_path}/{engine.name}"
        if not os.path.exists(engine_dir):
            os.makedirs(engine_dir)

        if engine.name == "openai_compatible":
            provider_dir = f"{engine_dir}/{engine.config.provider}"
            if not os.path.exists(provider_dir):
                os.makedirs(provider_dir)
            save_json_output_path = f"{provider_dir}/{engine.config.tokenizer.replace('/', '_')}.jsonl"
        else:
            save_json_output_path = f"{engine_dir}/{id}.jsonl"

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
        print(f"# Running Task {task}")
        task_outputs = []
        dataset = Dataset(DatasetConfig(task, limit=limit))
        for messages, schema in tqdm(
            dataset.iter(mf),
            total=safe_min(len(dataset), limit),
            desc=task,
            file=sys.stdout,
        ):
            # with disable_print(): # TODO uncomment later
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
