"""Utility class."""

import os
import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils


def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_dbr_host() -> str:
    """Retrieve the Databricks workspace URL.

    This function obtains the workspace URL from Spark configuration.

    :return: The Databricks workspace URL as a string.
    :raises ValueError: If not running in a Databricks environment.
    """
    ws = WorkspaceClient()
    return ws.config.host


def get_widget(name: str, default: str | None = None) -> str | None:
    """Get a Databricks widget value with a fallback default.

    :param name: Widget name.
    :param default: Default value if widget is not set.
    :return: Widget value or default.
    """

    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


def get_job_parameter(name: str, default: str | None = None) -> str | None:
    """Get a Databricks job parameter from widgets or CLI args.

    Supports both ``notebook_task`` widgets and ``spark_python_task`` parameters
    passed as ``--name value`` or ``--name-with-dashes value``.
    """

    option_names = [f"--{name}"]
    dashed_name = name.replace("_", "-")
    if dashed_name != name:
        option_names.append(f"--{dashed_name}")

    for index, arg in enumerate(sys.argv[:-1]):
        if arg in option_names:
            return sys.argv[index + 1]

        for option_name in option_names:
            prefix = f"{option_name}="
            if arg.startswith(prefix):
                return arg[len(prefix) :]

    value = get_widget(name)
    if value is not None:
        return value

    return default
