import numpy as np
from uuid import UUID
from json import loads
from typing import List, Tuple
from ipaddress import IPv4Address, IPv6Address
from jsonschema import Draft202012Validator, FormatChecker, SchemaError

from core.utils import bootstrap
from core.types import (
    Schema,
    CompileStatusCode,
    GenerationOutput,
    AggregatedPerfMetrics,
    Metric,
)


def is_json_schema_valid(schema: Schema):
    try:
        Draft202012Validator.check_schema(schema)
        return True
    except SchemaError:
        return False


format_checker = FormatChecker()


@format_checker.checks("ipv4")
def ipv4_check(value):
    IPv4Address(value)


@format_checker.checks("ipv6")
def ipv6_check(value):
    IPv6Address(value)


@format_checker.checks("uuid")
def uuid_check(value):
    UUID(value)


def validate_json_schema(instance: Schema, schema: Schema) -> bool:
    if not is_json_schema_valid(schema):
        return False
    validator = Draft202012Validator(schema, format_checker=format_checker)
    try:
        validator.validate(instance)

    # we catch all exceptions include ValidationError and Error from extension validators
    except Exception:
        return False
    return True


def evaluate(
    outputs: List[GenerationOutput],
) -> Tuple[Metric, Metric, Metric, AggregatedPerfMetrics, Metric, List[GenerationOutput]]:
    output_tokens_list = []
    declared_coverage_list = []
    empirical_coverage_list = []
    evaluated_outputs = []

    for generation_output in outputs:
        generation = generation_output.generation
        schema = generation_output.schema

        if schema is None or generation is None:
            generation_output.metadata.failure=True
            generation_output.metadata.failure_type="Empty generation or schema"
            evaluated_outputs.append(generation_output)
            continue

        if generation_output.metadata.compile_status.code == CompileStatusCode.OK:
            declared_coverage_list.append(1)
        else:
            declared_coverage_list.append(0)

        try:
            json_object = loads(generation)
        except Exception:
            empirical_coverage_list.append(0)
            generation_output.metadata.failure=True
            generation_output.metadata.failure_type="Generation is not json parsable"
            evaluated_outputs.append(generation_output)
            continue

        if not validate_json_schema(json_object, schema):
            empirical_coverage_list.append(0)
            generation_output.metadata.failure=True
            generation_output.metadata.failure_type="Generated json is not instance of the provided schema"
            evaluated_outputs.append(generation_output)
            continue

        empirical_coverage_list.append(1)
        generation_output.metadata.failure=False
        evaluated_outputs.append(generation_output)
        output_tokens_list.append(generation_output.token_usage.output_tokens)

    ttft_list = [
        generation_output.perf_metrics.ttft
        for generation_output in outputs
        if generation_output.perf_metrics.ttft is not None
    ]
    tpot_list = [
        generation_output.perf_metrics.tpot
        for generation_output in outputs
        if generation_output.perf_metrics.tpot is not None
    ]
    tgt_list = [
        generation_output.perf_metrics.tgt
        for generation_output in outputs
        if generation_output.perf_metrics.tgt is not None
    ]
    gct_list = [
        generation_output.perf_metrics.gct
        for generation_output in outputs
        if generation_output.perf_metrics.gct is not None
    ]

    compliance_list = [
        ec for ec, dc in zip(empirical_coverage_list, declared_coverage_list) if dc == 1
    ]

    dc_mean_list = bootstrap(declared_coverage_list, np.mean)
    ec_mean_list = bootstrap(empirical_coverage_list, np.mean)
    c_mean_list = bootstrap(compliance_list, np.mean)

    return (
        Metric.from_values(dc_mean_list),
        Metric.from_values(ec_mean_list),
        Metric.from_values(c_mean_list),
        AggregatedPerfMetrics(
            ttft=Metric.from_values(ttft_list),
            tpot=Metric.from_values(tpot_list),
            tgt=Metric.from_values(tgt_list),
            gct=Metric.from_values(gct_list),
        ),
        Metric.from_values(output_tokens_list),
        evaluated_outputs
    )
