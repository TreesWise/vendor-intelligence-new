import time
import logging
import threading
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
    _lock = threading.Lock()  # Lock to ensure thread-safe access

    def __new__(cls):
        """Create or return the singleton instance of the database connection."""
        with cls._lock:  # Ensure only one thread initializes the instance
            if cls._instance is None:
                logging.info("Creating new SQLDatabase instance...")
                start_time = time.time()
                try:
                    cls._instance = SQLDatabase.from_databricks(
                        catalog=catalog,
                        schema=schema,
                        api_token=api_token,
                        host=host,
                        warehouse_id=warehouse_id
                    )
                    end_time = time.time()
                    logging.info(f"SQLDatabase instance created successfully in {end_time - start_time} seconds.")
                except Exception as e:
                    logging.error("Error creating SQLDatabase instance:", exc_info=True)
                    raise RuntimeError("Failed to initialize SQLDatabase")
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Return the existing singleton instance."""
        if cls._instance is None:
            logging.warning("SQLDatabase instance is not created yet. Creating it now...")
            cls()  # Calls __new__() to create the instance
        return cls._instance

# # Ensure each worker initializes its database connection independently
# def initialize_database():
#     SingletonSQLDatabase.get_instance()
