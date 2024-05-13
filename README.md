# HPO-GLUE Usage

## Adding a new Optimizer

To implement an optimizer, you need to create a class that inherits from `Optimizer` in hpo_glue/glu.py and implements the `__init__()`, `ask()` and `tell()` methods.

Depending on the features supported, the Optimizer should set the following class variables to True:
`supports_manyfidelity`, `supports_multifidelity`, `supports_multiobjective`, `supports_tabular`.

#### Optimizer member functions:

| Function | Description |
| --- | --- |
| `__init()__` | accepts the `problem_statement` and `working_directory` arguments. |
| `ask()` | returns a `Query` object containing a `Config` and a `fidelity`. |
| `tell()` | accepts a `Result` object. |




## Adding a new Benchmark


### Tabular Benchmark

Tabular Benchmarks are required to return an instance of the `TabularBenchmark` class in `hpo_glue/benchmarks/tabular.py`. \
The `TabularBenchmark` class has the following data members that need to be set:

| Data Member | Description |
| --- | --- |
| `name`| Name of the benchmark |
| `table`| A pandas Dataframe containing the benchmark data |
| `id_key`| The name of the column containing the unique identifier for each Config |
| `config_keys`| A list of column names containing the hyperparameters for each Config |
| `result_keys`| A list of column names containing the results for each Config |
| `fidelity_keys`| (*Optional*) An str or a list of str denoting the column name(s) containing the fidelity for each Config |
| `remove_constants`| (*Optional*) A boolean denoting whether to remove constant columns from the table |
| `time_budget`| (*Optional*) An str denoting the column name containing the time budget for each Config |


### Surrogate Benchmark

Surrogate Benchmarks are required to return an instance of the `SurrogateBenchmark` class in `hpo_glue/benchmarks/surrogate.py`. \
The `SurrogateBenchmark` class has the following data members that need to be set:

| Data Member | Description |
| --- | --- |
| `name`| Name of the benchmark |
| `config_space`| A `ConfigSpace` object containing the hyperparameter space of the benchmark |
| `fidelity_space`| (*Optional*) A list of `int` or `float` containing the fidelity space of the benchmark |
| `query_function`| A function that takes a `Query` object and returns a `Result` object |
| `benchmark` | The actual benchmark object that is being wrapped |
| `time_budget`| (*Optional*) An str denoting the column name containing the time budget for each Config |


### get_benchmark()

Import and add the benchmark to the `BENCHMARK_FACTORIES` dict in `hpo_glue/benchmarks/benchmarks.py`.




## Using GLUE

`GLUE` is the main class that is used to run the HPO-GLUE package.\
`GLUE` class variable: `root`: (*defaults to the current working directory*) sets the path to the root directory of the project.

`GLUE`'s `run()` method is used to initialize Optimizers using a given `ProblemStatement`, asks Optimizers for new Configs and queries the Benchmark with the returned Configs. The `run()` method accepts the following arguments:

| Argument | Description |
| --- | --- |
| `problem_statement`| A `ProblemStatement` object containing the problem statement |
| `optimizer`| A reference to the Optimizer class to be used |
| `benchmark`| A `Benchmark` object containing the benchmark - either a `TabularBenchmark` or a `SurrogateBenchmark` |
| `save_dir` | Path to the directory where the results should be saved |
| `seed` | (*Optional*) Seed to be used for reproducibility |


To use `GLUE`, a `ProblemStatement` object needs to be created. This object contains the following data members:

| Data Member | Description |
| --- | --- |
|`name`| Name of the problem statement |
|`config_space`| A `ConfigSpace` object containing the hyperparameter space of the benchmark to be used |
|`fidelity_space`| (*Optional*) A list of `int` or `float` containing the fidelity space of the benchmark to be used -> No fidelity space indicates no multi-fidelity |
| `result_keys`| An str or a list of str denoting the objective(s) to be optimized |
| `budget_type`| An str denoting the type of budget to be used. Currently supported types are `n_trials`, `time_budget` and `fidelity_budget`|
| `budget`| An int or a float denoting the budget to be used |
| `fidelity_keys`| (*Optional*) An str or a list of str denoting the fidelity type to be used from the benchmark |
| `minimize`| (*defaults to True*) A boolean or a list of boolean denoting whether the objective(s) should be minimized or maximized. |


## Example Usage

The following example shows the simplest way to use GLUE with the `SMAC` Optimizer and the `YAHPO-Gym LCBench` Surrogate Benchmark:

```python

from hpo_glue.glu import ProblemStatement, GLUE, GLUEReport
from optimizers.smac import SMAC_Optimizer
from benchmarks.benchmarks import get_benchmark

# Get the benchmark
benchmark = get_benchmark(name = "yahpo",
                            benchmark_name = benchmark_name,
                            task_id = task_id,
                            datadir = datadir)

# Get the optimizer
optimizer = SMAC_Optimizer

# Create the problem statement
ps = ProblemStatement(
    name = "SMAC_on_Yahpo_Gym_LCBench_TestCE",
    config_space = benchmark.config_space,
    fidelity_space = benchmark.fidelity_space,
    result_keys = "test_cross_entropy",
    budget_type = "n_trials",
    budget = 100,
    fidelity_keys = "epoch",
    minimize = True)

# Run the optimizer on the benchmark using GLUE
glu_report = GLUE.run(problem_statement = ps,
                        optimizer = optimizer,
                        benchmark = benchmark,
                        save_dir = './results')

```
