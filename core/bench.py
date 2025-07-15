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

    if not isinstance(messages_formatter, list):
        messages_formatter = [messages_formatter] * len(tasks)

    all_outputs = []
    for task, mf in zip(tasks, messages_formatter):
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
        all_outputs.append(task_outputs)

    compliance = []
    perf_metrics = []
    output_tokens = []
    declared_coverage = []
    empirical_coverage = []
    all_evaluated_outputs = []
    for outputs in all_outputs:
        dc, ec, cl, pm, ot , evaluated_outputs= evaluate(outputs)

        compliance.append(cl)
        perf_metrics.append(pm)
        declared_coverage.append(dc)
        empirical_coverage.append(ec)
        output_tokens.append(ot)

        all_evaluated_outputs.append(evaluated_outputs)
        
    print_scores(
        declared_coverage,
        empirical_coverage,
        compliance,
        perf_metrics,
        output_tokens,
        tasks,
    )

    if save_outputs:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(f"{output_path}/{engine.name}"):
            os.makedirs(f"{output_path}/{engine.name}")
            
        if engine.name == "openai_compatible":
            if not os.path.exists(f"{output_path}/{engine.name}/{engine.config.provider}"):
                os.makedirs(f"{output_path}/{engine.name}/{engine.config.provider}")
            save_json_output_path = f"{output_path}/{engine.name}/{engine.config.provider}/{engine.config.tokenizer.replace('/','_')}.jsonl"
        else:
            save_json_output_path = f"{output_path}/{engine.name}/{id}.jsonl"

        with open(save_json_output_path, "w") as f:
            f.write(
                f"{dumps({'engine': engine.name, 'engine_config': asdict(engine.config)})}\n"
            )

            for outputs in all_evaluated_outputs:
                for output in outputs:
                    f.write(f"{dumps(asdict(output))}\n")

        print(f"Outputs saved to {save_json_output_path}")

    if close_engine:
        engine.close()

    return all_outputs
