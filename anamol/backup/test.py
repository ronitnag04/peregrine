# %%
import pandas as pd
import numpy as np

# %%
# Load the array
data = np.load("rob.npy")

print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(data[5])


# %%
data_raw = pd.read_csv(
    "traces/trace.csv",
    dtype={
        "IP": "string",
        "Assembly": "string",
        "Category": "string",
        "Opcode": "string",
        "Branch Type": "string",
        "Instruction Sync": "string",  # will convert to bool below
        # list-like columns read as string then split later
        "Read Registers": "string",
        "Write Registers": "string",
        "Register Dependent IPs": "string",
        "Read Addresses": "string",
        "Write Addresses": "string",
        "Memory Dependent IPs": "string",
    },
    keep_default_na=True,
)


# %%
data = data_raw.copy()
multicols = [
    "Read Registers",
    "Write Registers",
    "Register Dependent IPs",
    "Read Addresses",
    "Write Addresses",
    "Memory Dependent IPs",
]

for col in multicols:
    data[col] = data[col].str.split(";").fillna("")

# Add size and cumulative count for each column
for col in multicols:
    # Size: number of elements in each row
    data[f"{col} - size"] = data[col].apply(len).astype("UInt32")
    data[f"{col} - index"] = (
        data[f"{col} - size"].cumsum().shift(fill_value=0).astype("UInt32")
    )
    # Cumulative count: running total of size (before adding current row)

# %%
data

# %%
multi_dfs = {}
for col in multicols:
    multi_dfs[col] = data[col][data[col].str.len() > 3].explode()

data = data.drop(columns=multicols)

# %%
data.dtypes
# %%
data

# %%
