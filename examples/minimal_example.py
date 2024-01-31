from hpo_glue.glu import ProblemStatement, Problem, GLUE
from optimizers.smac import SMAC_Hyperband
from benchmarks.benchmarks import get_benchmark

# Get the benchmark
benchmark = get_benchmark(name = "yahpo",
                          benchmark_name = "lcbench",
                          task_id = "3945", 
                          datadir = "./data")

# Get the optimizer
optimizer = SMAC_Hyperband

problem_statement = ProblemStatement(
    benchmark = benchmark,
    optimizer = optimizer,
    hyperparameters = {
        'eta' : 2,
    }
)

problem = Problem(
    problem_statement = problem_statement,
    objectives = "test_balanced_accuracy",
    minimize = False,
    fidelities = benchmark.fidelity_keys
)

GLUE.run(
    problem = problem,
    exp_dir = "./results/min_ex_test",
    budget_type = "fidelity_budget",
    budget = 1000,
    seed = 80
)