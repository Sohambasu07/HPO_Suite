from __future__ import annotations
import inspect

from hpo_glue.glu import Optimizer, Benchmark, Problem, TabularBenchmark, SurrogateBenchmark
from benchmarks.benchmarks import get_all_benchmarks, get_benchmark
from optimizers.optimizers import get_all_optimizers, get_optimizer

def problem_generator(
        optimizer: Optimizer | None = None,
        benchmark: Benchmark | None = None
    ) -> list[Problem]:
        """Generates a list of problems based on the optimizer(s) and/or benchmark(s) selected"""

        options = {
            "Benchmarks": {},
            "Optimizers": {},
            "Objectives": {},
            "Fidelities": {},
            "Problems": []
        }

        if optimizer is None and benchmark is None:
            raise ValueError("Must specify either optimizer or benchmark!")
        
        elif optimizer is not None and benchmark is not None:

            if isinstance(benchmark, TabularBenchmark):
                assert optimizer.supports_tabular
            
        elif optimizer is not None:

            options["Optimizers"][optimizer.name] = optimizer
            benchmarks = get_all_benchmarks()

            # Adding benchmarks that are compatible with the optimizer

            for name, benchmark in benchmarks:
                if isinstance(benchmark, SurrogateBenchmark):
                    options["Benchmarks"][name] = get_benchmark(name)
                elif optimizer.supports_tabular:
                    options["Benchmarks"][name] = get_benchmark(name)

            print(f"Based on the Optimizer selected - {optimizer.name}, the following benchmarks are available:")
            print(options.get("Benchmarks").keys())

            # Adding Objectives for each benchmark

            print("Use default objectives for benchmarks? Type Y or N")
            use_default = input()

            if use_default == "Y":
                for _, bench in options.get("Benchmarks"):
                    options["Objectives"][bench] = benchmarks.get(bench).default_objective
            else:
                print("Select objectives for each benchmark. Type -> str")
                for _, bench in options.get("Benchmarks"):
                    print(f"Select objectives for {bench}")
                    print(f"Available objectives for {bench}: {bench.result_keys}")
                    obj = input()
                    if obj not in benchmarks.get(bench).result_keys:
                        raise ValueError(f"{obj} is not a valid objective for {bench}")
                    options["Objectives"][bench] = obj

            # Adding Fidelities for each benchmark

            for _, bench in options.get("Benchmarks"):
                fids = benchmarks.get(bench).fidelity_keys
                if isinstance(fids, str):
                    options["Fidelities"][bench] = fids
                elif isinstance(fids, list):
                    options["Fidelities"][bench] = fids[0]            
            
            # Generating problems:

            for name, bench in options.get("Benchmarks"):
                options["Problems"].append(Problem(
                    name = name + "_" + optimizer.name,
                    optimizer = optimizer,
                    benchmark = bench,
                    config_space = benchmarks.get(bench).config_space,
                    fidelity_space = benchmarks.get(bench).fidelity_space,
                    result_keys = options.get("Objectives").get(bench),
                    fidelity_keys = options.get("Fidelities").get(bench),
                    minimize = bench.minimize_default
                ))

            print("Generated the following problems:")
            print(options.get("Problems"))

            return options.get("Problems")
        
        elif benchmark is not None:
            options["Benchmarks"][benchmark.name] = benchmark
            optimizers = get_all_optimizers()

            # Adding optimizers that are compatible with the benchmark

            # default_params = ["self", "problem", "working_directory", "seed"]
            # required_params = {}

            for name, optimizer in optimizers:
                if isinstance(benchmark, SurrogateBenchmark):
                    # for params in inspect.signature(
                    #     optimizer.__init__).parameters.keys() not in default_params:
                    options["Optimizers"][name] = get_optimizer(name)
                elif optimizer.supports_tabular:
                    options["Optimizers"][name] = get_optimizer(name)

            print(f"Based on the Benchmark selected - {benchmark.name}, the following optimizers are available:")
            print(options.get("Optimizers").keys())

            # Adding Objectives for the benchmark

            print("Use default objective for the benchmark? Type Y or N")
            use_default = input()

            obj = benchmark.default_objective
            if use_default != "Y":
                print("Select objective for the benchmark. Type -> str")
                print(f"Available objectives for {benchmark.name}: {benchmark.result_keys}")
                obj = input()
                if obj not in benchmark.result_keys:
                    raise ValueError(f"{obj} is not a valid objective for {benchmark.name}")

            # Adding Fidelities for the benchmark
                
            fids = benchmark.fidelity_keys
            if isinstance(fids, list):
                fids = fids[0]

            # Generating problems:
                
            for name, optimizer in options.get("Optimizers"):
                fidelity = fids
                fidelity_space = benchmark.fidelity_space
                if optimizer.supports_multifidelity is False:
                    fidelity = None
                    fidelity_space = None                    
                options["Problems"].append(Problem(
                    name = benchmark.name + "_" + optimizer.name,
                    optimizer = optimizer,
                    benchmark = benchmark,
                    config_space = benchmark.config_space,
                    fidelity_space = fidelity_space,
                    result_keys = obj,
                    fidelity_keys = fidelity,
                    minimize = benchmark.minimize_default
                ))

            print("Generated the following problems:")
            print(options.get("Problems"))

            return options.get("Problems")
                
                