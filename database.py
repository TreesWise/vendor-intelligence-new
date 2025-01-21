
import logging
import threading
import requests
import time
from langchain_community.utilities.sql_database import SQLDatabase
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone
import datetime
from functools import lru_cache
import os

# Fetch credentials from environment variables
API_TOKEN = os.getenv("API_TOKEN")
HOST = os.getenv("HOST")
WAREHOUSE_ID = os.getenv("WAREHOUSE_ID")

# Databricks connection details
CATALOG = "hive_metastore"
SCHEMA = "Common"


# Configure logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Using Databricks host: {HOST}")

# Helper to check warehouse status with caching
@lru_cache(maxsize=1)
def is_warehouse_running():
    url = f"https://{HOST}/api/2.0/sql/warehouses/{WAREHOUSE_ID}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            state = response.json().get("state")
            logging.info(f"Warehouse state: {state}")
            return state == "RUNNING"
        else:
            logging.error(f"Failed to check warehouse state: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Error while checking warehouse state: {e}")
    return False

def start_databricks_warehouse():
    if is_warehouse_running():
        logging.info("Warehouse is already running.")
        return
    url = f"https://{HOST}/api/2.0/sql/warehouses/{WAREHOUSE_ID}/start"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            logging.info("Databricks SQL Warehouse is starting...")
            is_warehouse_running.cache_clear()  # Clear cache to refresh state
        else:
            logging.error(f"Failed to start SQL Warehouse: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Error while starting warehouse: {e}")

def stop_databricks_warehouse():
    url = f"https://{HOST}/api/2.0/sql/warehouses/{WAREHOUSE_ID}/stop"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            logging.info("Databricks SQL Warehouse is stopping...")
            is_warehouse_running.cache_clear()  # Clear cache to refresh state
        else:
            logging.error(f"Failed to stop SQL Warehouse: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Error while stopping warehouse: {e}")

class SingletonSQLDatabase:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    logging.info("Checking and starting Databricks warehouse before SQL connection...")
                    start_databricks_warehouse()
                    logging.info("Initializing SQLDatabase instance...")
                    cls._instance = cls._initialize_instance()
        return cls._instance

    @classmethod
    def _initialize_instance(cls):
        try:
            return SQLDatabase.from_databricks(
                catalog=CATALOG,
                schema=SCHEMA,
                api_token=API_TOKEN,
                host=HOST,
                warehouse_id=WAREHOUSE_ID,
            )
        except Exception as e:
            logging.error("Failed to initialize SQLDatabase:", exc_info=True)
            raise RuntimeError("Failed to initialize SQLDatabase") from e

    @classmethod
    def get_instance(cls):
        return cls.__new__(cls)

# Scheduler setup
local_timezone = timezone('Asia/Kolkata')
scheduler = BackgroundScheduler(timezone=local_timezone)

# Start at 7:00 AM
scheduler.add_job(start_databricks_warehouse, 'cron', hour=7, minute=0)
logging.info("Scheduled Databricks SQL Warehouse to start at 7:00 AM")

# # Stop at 7:00 PM
# scheduler.add_job(stop_databricks_warehouse, 'cron', hour=19, minute=0)
# logging.info("Scheduled Databricks SQL Warehouse to stop at 7:00 PM")
# Stop at 3:10 PM
scheduler.add_job(stop_databricks_warehouse, 'cron', hour=15, minute=10)
logging.info("Scheduled Databricks SQL Warehouse to stop at 3:10 PM")

def initialize_scheduler():
    try:
        scheduler.start()
        logging.info("Scheduler started successfully.")
    except Exception as e:
        logging.error(f"Scheduler failed to start: {e}")

def keep_alive():
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logging.info("Scheduler shut down gracefully.")

# Start scheduler and keep alive
threading.Thread(target=initialize_scheduler, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()

logging.info(f"Current time: {datetime.datetime.now(local_timezone)}")
