from benchmarks.benchmarks import get_benchmark
from optimizers.dehb import DEHB_Optimizer

from hpo_glue.glu import GLUE, Problem, ProblemStatement

# Get the benchmark
benchmark = get_benchmark(name="yahpo", benchmark_name="lcbench", task_id="3945", datadir="./data")

# Get the optimizer
# optimizer = SMAC_Hyperband
# optimizer = SyneTuneOptimizer
optimizer = DEHB_Optimizer

problem_statement = ProblemStatement(
    benchmark=benchmark,
    optimizer=optimizer,
    hyperparameters={
        "eta": 2,
    },
)

problem = Problem(
    problem_statement=problem_statement,
    objective="test_cross_entropy",
    minimize=True,
    fidelity_key=benchmark.fidelity_keys,
)

GLUE.run(
    problem=problem,
    exp_dirname="./results/min_ex_test",
    budget_type="fidelity_budget",
    budget=1000,
    seed=80,
)
