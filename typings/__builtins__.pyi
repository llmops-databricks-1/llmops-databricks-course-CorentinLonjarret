from databricks.sdk.runtime import *  # noqa: F403
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import udf as U
from pyspark.sql.session import SparkSession

udf = U
spark: SparkSession
sc = spark.sparkContext
sqlContext: SQLContext
sql = sqlContext.sql
table = sqlContext.table
getArgument = dbutils.widgets.getArgument  # noqa: F405

def displayHTML(html: str) -> None: ...
def display(input: object = None, *args: object, **kwargs: object) -> None: ...
