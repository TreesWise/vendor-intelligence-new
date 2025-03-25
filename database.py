import logging
import threading
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv
import os

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
HOST = os.getenv("HOST")
WAREHOUSE_ID = os.getenv("WAREHOUSE_ID")
CATALOG = "hive_metastore"
SCHEMA = "Common"

logging.basicConfig(level=logging.INFO)
logging.info(f"Using Databricks host: {HOST}")


class SingletonSQLDatabase:
    """Thread-safe Singleton for managing a shared SQLDatabase instance that auto-refreshes on failure."""
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def _create_instance(cls):
        from sqlalchemy.engine import URL, create_engine

        try:
            # Build Databricks SQLAlchemy URL
            url = URL.create(
                "databricks+connector",
                host=HOST,
                token=API_TOKEN,
                catalog=CATALOG,
                schema=SCHEMA,
                warehouse_id=WAREHOUSE_ID,
            )

            # Enable pool_pre_ping to avoid stale connections
            engine = create_engine(url, pool_pre_ping=True)

            logging.info("Successfully created SQLDatabase instance.")
            return SQLDatabase(engine=engine)

        except Exception as e:
            logging.error("Failed to create SQLDatabase instance.", exc_info=True)
            raise RuntimeError("Failed to initialize SQLDatabase") from e

    @classmethod
    def get_instance(cls):
        """Returns a valid SQLDatabase instance, refreshing if needed."""
        with cls._lock:
            if cls._instance is None:
                logging.info("Initializing new SQLDatabase instance...")
                cls._instance = cls._create_instance()
            else:
                # Validate connection with a quick query
                try:
                    cls._instance.run("SELECT 1")
                except OperationalError as e:
                    logging.warning("Stale or broken SQL connection detected. Reinitializing...")
                    cls._instance = cls._create_instance()
                except Exception as e:
                    logging.error("Unexpected DB error during health check", exc_info=True)
                    raise
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Forcefully reset the instance (manual override)."""
        with cls._lock:
            logging.info("Resetting SQLDatabase singleton instance.")
            cls._instance = None
