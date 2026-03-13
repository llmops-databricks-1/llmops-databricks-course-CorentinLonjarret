"""ArXiv data ingestion — fetch papers and persist them as a Delta table in Unity Catalog."""

from datetime import datetime

import arxiv
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType

from llmops_databricks_course_CorentinLonjarret.config import ProjectConfig

ARXIV_SCHEMA = StructType(
    [
        StructField("arxiv_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("authors", ArrayType(StringType()), True),
        StructField("summary", StringType(), True),
        StructField("published", LongType(), True),
        StructField("updated", StringType(), True),
        StructField("categories", StringType(), True),
        StructField("pdf_url", StringType(), True),
        StructField("primary_category", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
        StructField("processed", LongType(), True),
        StructField("volume_path", StringType(), True),
    ]
)


class ArxivDataIngester:
    """Fetch arXiv paper metadata and persist it as a Delta table in Unity Catalog."""

    def __init__(self, spark: SparkSession, cfg: ProjectConfig) -> None:
        self.spark = spark
        self.cfg = cfg
        self._table_path = f"{cfg.catalog}.{cfg.db_schema}.{cfg.table}"

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------
    def check_schema(self) -> None:
        """Verify that the target Unity Catalog schema exists.

        Raises:
            Exception: if the schema cannot be found or accessed.
        """
        try:
            self.spark.sql(f"USE {self.cfg.catalog}.{self.cfg.db_schema}")
        except Exception:
            logger.error(
                f"Schema {self.cfg.catalog}.{self.cfg.db_schema} does not exist. "
                "Please ask your Databricks admin to create the schema, "
                "or update your config to use an existing schema."
            )
            raise
        logger.info(f"Schema {self.cfg.catalog}.{self.cfg.db_schema} ready")

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    def fetch_papers(
        self,
        query: str = "cat:cs.AI OR cat:cs.LG",
        max_results: int = 50,
    ) -> list[dict]:
        """Fetch arXiv papers using the arXiv API.

        Args:
            query: arXiv search query.
            max_results: Maximum number of papers to fetch.

        Returns:
            List of paper metadata dictionaries.
        """
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        for result in client.results(search):
            paper = {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": int(result.published.strftime("%Y%m%d%H%M")),
                "updated": result.updated.isoformat() if result.updated else None,
                "categories": ", ".join(result.categories),
                "pdf_url": result.pdf_url,
                "primary_category": result.primary_category,
                "ingestion_timestamp": datetime.now().isoformat(),
                "processed": None,
                "volume_path": None,
            }
            papers.append(paper)

        logger.info(f"Fetched {len(papers)} papers")
        if papers:
            logger.info(f"Title: {papers[0]['title']}")
            logger.info(f"Authors: {papers[0]['authors']}")
            logger.info(f"arXiv ID: {papers[0]['arxiv_id']}")
            logger.info(f"PDF URL: {papers[0]['pdf_url']}")
        return papers

    # ------------------------------------------------------------------
    # Delta table creation
    # ------------------------------------------------------------------
    def create_table(self, papers: list[dict]) -> DataFrame:
        """Create a Spark DataFrame from paper dicts and write it as a Delta table.

        Args:
            papers: List of paper metadata dictionaries (as returned by fetch_papers).

        Returns:
            The DataFrame that was written.
        """
        df = self.spark.createDataFrame(papers, schema=ARXIV_SCHEMA)

        df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(self._table_path)

        logger.info(f"Created Delta table: {self._table_path}")
        logger.info(f"Records: {df.count()}")
        return df

    # ------------------------------------------------------------------
    # Load & stats
    # ------------------------------------------------------------------

    def load_table(self) -> DataFrame:
        """Read the Delta table back from Unity Catalog.

        Returns:
            DataFrame with all rows from the table.
        """
        df = self.spark.table(self._table_path)
        logger.info(f"Table: {self._table_path}")
        logger.info(f"Total papers: {df.count()}")
        return df
