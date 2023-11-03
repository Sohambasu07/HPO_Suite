import mfpbench
from pathlib import Path
import os

data_dir = Path("./data") / "lcbench-tabular"


lcbench_adult_tabular = mfpbench.get("lcbench_tabular", task_id="adult", datadir=data_dir)
table = lcbench_adult_tabular.table
print(table.columns)

# print(lcbench_adult_tabular.query({"id": 0}, at=26))
print(table.loc[('77', 14)])
# print(table.index.get_level_values("epoch").unique()[-1])


# count = 0
# for id, data in table.iterrows():
#     print(id)
#     # print(dict(zip(["time", "momentum"], data.get(["time", "momentum"]).values)))
#     print(data)
#     print(data.get(["time", "momentum"]).to_dict())
#     print(type(data))
#     # print(config)
#     count+=1
#     if count>0:
#         break



# print(table.idxmax())
# print(type(lcbench_adult_tabular))
# config = lcbench_adult_tabular.sample()
# print(lcbench_adult_tabular.query(config, at=22))