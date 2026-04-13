import mlflow
from mlflow.models import ModelConfig

from arxiv_curator.agent import ArxivAgent

config = ModelConfig(
    development_config={
        "catalog": "PLACEHOLDER",
        "schema": "PLACEHOLDER",
        "genie_space_id": "PLACEHOLDER",
        "system_prompt": "PLACEHOLDER",
        "llm_endpoint": "PLACEHOLDER",
        "lakebase_project_id": "PLACEHOLDER",
    }
)

agent = ArxivAgent(
    llm_endpoint=config.get("llm_endpoint"),
    system_prompt=config.get("system_prompt"),
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    genie_space_id=config.get("genie_space_id"),
    lakebase_project_id=config.get("lakebase_project_id"),
)
mlflow.models.set_model(agent)  # type: ignore
