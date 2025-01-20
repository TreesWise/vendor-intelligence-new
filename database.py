
import logging
import threading
import requests
import time
from langchain_community.utilities.sql_database import SQLDatabase
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone
import datetime
import signal


# Fetch credentials from environment variables
API_TOKEN = os.getenv("API_TOKEN")
HOST = os.getenv("HOST")
WAREHOUSE_ID = os.getenv("WAREHOUSE_ID")

# Databricks connection details
CATALOG = "hive_metastore"
SCHEMA = "Common"


# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logging.info(f"Using Databricks host: {HOST}")


def start_databricks_warehouse():
    url = f"https://{HOST}/api/2.0/sql/warehouses/{WAREHOUSE_ID}/start"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            logging.info("Databricks SQL Warehouse is starting...")
        else:
            logging.error(f"Failed to start SQL Warehouse: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Exception during start_databricks_warehouse: {e}")


def stop_databricks_warehouse():
    url = f"https://{HOST}/api/2.0/sql/warehouses/{WAREHOUSE_ID}/stop"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            logging.info("Databricks SQL Warehouse is stopping...")
        else:
            logging.error(f"Failed to stop SQL Warehouse: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Exception during stop_databricks_warehouse: {e}")


class SingletonSQLDatabase:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    logging.info("Starting Databricks warehouse before SQL connection...")
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

# Schedule jobs
scheduler.add_job(start_databricks_warehouse, 'cron', hour=7, minute=0)
logging.info("Scheduled Databricks SQL Warehouse to start daily at 7:00 AM")

scheduler.add_job(stop_databricks_warehouse, 'cron', hour=19, minute=0)
logging.info("Scheduled Databricks SQL Warehouse to stop daily at 7:00 PM")


def run_scheduler():
    try:
        scheduler.start()
        logging.info("Scheduler started successfully.")
    except Exception as e:
        logging.error(f"Scheduler failed to start: {e}")


# Run scheduler in a separate thread
threading.Thread(target=run_scheduler, daemon=True).start()

# Debugging: Print scheduled jobs
for job in scheduler.get_jobs():
    logging.info(f"Scheduled Job: {job}")

# Log current time to check timezone
current_time = datetime.datetime.now(local_timezone)
logging.info(f"Current time: {current_time}")


# Graceful shutdown handling
def signal_handler(sig, frame):
    logging.info("Shutting down gracefully...")
    scheduler.shutdown()
    logging.info("Scheduler shut down successfully.")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Minimal FastAPI integration for testing
if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"message": "Hello, World!"}

    logging.info("Starting FastAPI server...")

    # Run FastAPI on a different port if needed
    uvicorn.run(app, host="127.0.0.1", port=8000)
