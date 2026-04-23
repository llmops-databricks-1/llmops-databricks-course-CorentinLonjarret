"""Configuration management for Arxiv Curator."""

import sys
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession


class ProjectConfig(BaseModel):
    """Project configuration model."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    llm_endpoint: str = Field(..., description="LLM endpoint name")
    papers_table: str = Field(..., description="Papers table name")
    parsed_table: str = Field(..., description="Parsed table name")
    chunks_table: str = Field(..., description="Chunks table name")
    index_table: str = Field(..., description="Index table name")
    aggregated_view: str = Field(..., description="Aggregated view name")
    embedding_endpoint: str = Field(..., description="Embedding endpoint name")
    vector_search_endpoint: str = Field(..., description="Vector search endpoint name")
    genie_space_id: str = Field(..., description="Genie space ID")
    warehouse_id: str | None = Field(None, description="SQL warehouse ID backing the Genie space")
    lakebase_project_id: str = Field(..., description="Lakebase project ID")
    experiment_name: str = Field(..., description="Experiment name")
    agent_name: str = Field(..., description="Registered model name")
    system_prompt: str = Field(
        default="You are a helpful AI assistant that helps users find and understand research papers.",
        description="System prompt for the agent",
    )
    arxiv_max_results_per_request: int = Field(..., description="Max results per arXiv API request")
    arxiv_end_date_request: str | None = Field(
        None, description="End date for arXiv request in YYYYMMDDHH format. None means current time."
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prod)

        Returns:
            ProjectConfig instance
        """
        if env not in ["prod", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prod', 'acc', or 'dev'")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")

        return cls(**config_data[env])

    @property
    def schema(self) -> str:  # type: ignore
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.schema}.{self.volume}"


def load_config(config_path: str = "project_config.yml", env: str = "dev") -> ProjectConfig:
    """Load project configuration.

    Args:
        config_path: Path to configuration file
        env: Environment name

    Returns:
        ProjectConfig instance
    """
    if not Path(config_path).is_absolute():
        config_path = str(resolve_project_path(config_path))

    return ProjectConfig.from_yaml(config_path, env)


def resolve_project_path(path: str) -> Path:
    """Resolve a project-relative path from the current runtime context.

    This supports local execution, notebooks, and Databricks ``spark_python_task``
    runs where ``__file__`` may not be defined.
    """

    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj

    search_roots: list[Path] = [Path.cwd()]
    if sys.argv and sys.argv[0]:
        try:
            search_roots.append(Path(sys.argv[0]).resolve().parent)
        except OSError:
            logger.warning(f"Could not resolve sys.argv[0]: {sys.argv[0]}")

    seen: set[Path] = set()
    for root in search_roots:
        current = root.resolve()
        for _ in range(5):
            if current in seen:
                break
            seen.add(current)

            candidate = current / path_obj
            if candidate.exists():
                return candidate

            if current.parent == current:
                break
            current = current.parent

    raise FileNotFoundError(f"Could not resolve project path: {path}")


def get_env(spark: SparkSession) -> str:
    """Get current environment from dbutils widget or CLI args, falling back to 'dev'.

    Returns:
        Environment name (dev, acc, or prod)
    """
    try:
        dbutils = DBUtils(spark)
        env = dbutils.widgets.get("env")
        logger.info(f"get_env: Retrieved environment from dbutils widget: {env}")
        return env
    except Exception:
        for index, arg in enumerate(sys.argv):
            if arg == "--env" and index + 1 < len(sys.argv):
                env = sys.argv[index + 1]
                logger.info(f"get_env: Retrieved environment from CLI args: {env}")
                return env

        logger.warning("get_env: Could not retrieve environment from dbutils or CLI args, falling back to 'dev'")
        return "dev"
