from dagster import Definitions, load_assets_from_modules
from rlhf_training.assets import data, model, mt_benchmark


model_assets = load_assets_from_modules([model], group_name="model")
data_assets = load_assets_from_modules([data], group_name="data")
mt_benchmark_assets = load_assets_from_modules([mt_benchmark], group_name="mt_benchmark")

defs = Definitions(assets=model_assets + data_assets + mt_benchmark_assets)
