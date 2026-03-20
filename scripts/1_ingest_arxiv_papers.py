"""Script d'orchestration — ingestion des papers arXiv vers Unity Catalog.

Usage:
    python scripts/1_ingest_arxiv_papers.py --root_path /path/to/repo --env dev
"""

from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.arxiv_ingestion import ArxivDataIngester
from arxiv_curator.config import get_env, load_config

# Init Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
logger.info(f"Loading configuration (env={env})")
cfg = load_config("project_config.yml", env)
logger.info(f"Configuration loaded: catalog={cfg.catalog} schema={cfg.schema_name} processed_table={cfg.table}")

# Init ArxivDataIngester
logger.info("Initializing ArxivDataIngester")
ingester = ArxivDataIngester(spark, cfg)

# Verify schema / Fetch arXiv papers / Create table
ingester.check_schema()
papers = ingester.fetch_papers(query="cat:cs.AI OR cat:cs.LG", max_results=50)
df_papers = ingester.create_table(papers)

# print statistics
logger.info("Sample records:")
df_papers.select("arxiv_id", "title", "primary_category", "published").show(5, truncate=50)

logger.info("Papers by primary category:")
df_papers.groupBy("primary_category").count().orderBy("count", ascending=False).show()
