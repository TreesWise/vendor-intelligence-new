import logging
from langchain_community.utilities.sql_database import SQLDatabase
import os
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env") 

databricks_api = os.getenv("api_token")
host = os.getenv("host")
warehouse_id = os.getenv("warehouse_id")

# Databricks connection details
# host = "adb-1987506542517093.13.azuredatabricks.net"
api_token = databricks_api
catalog = "hive_metastore"  # Catalog name
schema = "Common"  # Schema name
# warehouse_id = "3248efb5151bc56e"

logging.info(f"Using Databricks host: {host}")


class SingletonSQLDatabase:
    _instance = None

    def __new__(cls):
        """Create or return the singleton instance of the database connection."""
        if cls._instance is None:
            logging.info("Creating new SQLDatabase instance...")
            try:
                cls._instance = SQLDatabase.from_databricks(
                    catalog=catalog,
                    schema=schema,
                    api_token=api_token,
                    host=host,
                    warehouse_id=warehouse_id,
                )
                logging.info("SQLDatabase instance created successfully.")
            except Exception as e:
                logging.error("Error creating SQLDatabase instance:", exc_info=True)
                raise RuntimeError("Failed to initialize SQLDatabase")
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Return the existing singleton instance."""
        if cls._instance is None:
            logging.warning("SQLDatabase instance is not created yet. Creating it now...")
            cls()
        return cls._instance


# Eager initialization: create the connection as soon as the module is imported
SingletonSQLDatabase.get_instance()
