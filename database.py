import logging
from langchain_community.utilities.sql_database import SQLDatabase
import os

# Fetch credentials from environment variables
api_token = os.getenv("API_TOKEN")
host = os.getenv("HOST")
warehouse_id = os.getenv("WAREHOUSE_ID")

# Databricks connection details
catalog = "hive_metastore"  # Catalog name
schema = "Common"  # Schema name

logging.info(f"Using Databricks host: {host}")


class SingletonSQLDatabase:
    _instance = None

    def __new__(cls):
        """Create or return the singleton instance of the database connection."""
        if cls._instance is None:
            logging.info("Creating new SQLDatabase instance...")
            try:
                # Adjust timeout and other parameters as needed
                cls._instance = SQLDatabase.from_databricks(
                    catalog=catalog,
                    schema=schema,
                    api_token=api_token,
                    host=host,
                    warehouse_id=warehouse_id,
                    timeout=120  # Set timeout to 2 minutes (adjustable)
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
