"""Script d'orchestration — ingestion des papers arXiv vers Unity Catalog.

Usage:
    python scripts/1_ingest_arxiv_papers.py --root_path /path/to/repo --env dev
"""

from arxiv_curator.vector_search import VectorSearchManager
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.data_processor import DataProcessor

# Init Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
logger.info(f"Loading configuration (env={env})")

cfg = load_config("project_config.yml", env=env)
logger.info(f"Configuration loaded: catalog={cfg.catalog} schema={cfg.schema_name} processed_table={cfg.table}")

# Step 1: Process New Papers
logger.info("Processing new papers")
processor = DataProcessor(spark=spark, config=cfg)
processor.process_and_save()

# Step 2: Sync Vector Search Index
logger.info("Syncing vector search index")
vs_manager = VectorSearchManager(config=cfg)
vs_manager.sync_index()

logger.info("✓ Data processing pipeline complete!")
