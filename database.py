import time
import logging
import threading
import os
from langchain_community.utilities.sql_database import SQLDatabase

# Fetch credentials from environment variables
api_token = os.getenv("API_TOKEN")
host = os.getenv("HOST")
warehouse_id = os.getenv("WAREHOUSE_ID")

# Databricks connection details
catalog = "hive_metastore"
schema = "Common"

logging.info(f"Using Databricks host: {host}")

class SingletonSQLDatabase:
    _instance = None
    _lock = threading.Lock()  # Lock to ensure thread-safe access

    def __new__(cls):
        """Create or return the singleton instance of the database connection."""
        with cls._lock:  # Ensure only one thread initializes the instance
            if cls._instance is None:
                logging.info("Creating new SQLDatabase instance...")
                try:
                    # Establish the connection using the credentials
                    cls._instance = SQLDatabase.from_databricks(
                        catalog=catalog,
                        schema=schema,
                        api_token=api_token,
                        host=host,
                        warehouse_id=warehouse_id
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
            cls()  # Calls __new__() to create the instance if it doesn't exist
        return cls._instance
