#!/bin/bash

echo "🔄 Cleaning .pyc files..."
find . -type f -name "*.pyc" -delete

echo "🚀 Starting Gunicorn server..."
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
