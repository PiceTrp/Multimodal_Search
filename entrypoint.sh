#!/bin/bash

# Start FastAPI in the background
python -m api.app &

# Wait a bit for FastAPI to start
sleep 3

# Start Streamlit
streamlit run ui/streamlit_app.py