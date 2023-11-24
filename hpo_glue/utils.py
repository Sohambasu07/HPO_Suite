from pathlib import Path
import matplotlib.pyplot as plt

from hpo_glue.glu import GLUEReport

def plot_incumbents(report: GLUEReport, save_dir: Path, objective: str, min_budget: int, max_budget: int):
    """Plot the incumbent performance over time"""

    if isinstance(objective, list):
        raise NotImplementedError("Multiobjective not yet implemented")

    df_list = report.history[objective].astype(float).values

    if report.problem_statement.minimize is False:
        df_list *= -1

    costs = []
    budgets = []
    count = 0
    budget = 0
    best_cost = None
    for cost in df_list:               
        if best_cost is None or abs(cost) < abs(best_cost):
            best_cost = cost
            costs.append(best_cost)

            if report.problem_statement.budget_type == "n_trials":
                budgets.append(count)

            elif report.problem_statement.budget_type == "time_budget":

                # TODO: For now only works if the time_budget key is "time"
                if "time" not in report.history.columns.values:
                    raise NotImplementedError("Not the correct key for time budget")
                budget += float(report.history["time"].values[count])
                budgets.append(budget)

            elif report.problem_statement.budget_type == "fidelity_budget":
                if report.problem_statement.fidelity_space is None:
                    try:
                        raise NotImplementedError(
                            f"Problem Statement: {report.problem_statement.name}: "
                            "Plotting for non-multifidelity with fidelity budget has not been implemented yet!")
                    except Exception as e:
                        print(repr(e))
                        return                       
                budget += int(report.history["Fidelity"].values[count])
                budgets.append(budget)

        count += 1

    plt.figure(f"Trajectory for {report.problem_statement.name}")
    plt.plot(budgets, costs)
    plt.scatter(budgets, costs, marker="x", color="red")
    plt.xlim(min_budget, max_budget)
    plt.xlabel("Budget")
    plt.ylabel("Cost")
    plt.title("Trajectory over time")
    plt.savefig(save_dir / Path(report.problem_statement.name + "_trajectory.png"))
    plt.show()