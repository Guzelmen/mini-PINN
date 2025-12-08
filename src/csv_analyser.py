import pandas as pd
from utils import PROJECT_ROOT
filepath = PROJECT_ROOT / "wandb_csv_files/v2_newlr_ad_hardmode_residual.csv"
# read first 2 columns
data = pd.read_csv(filepath)
# print first 2 columns, every 10 rows
print(data.iloc[::10, 1])
