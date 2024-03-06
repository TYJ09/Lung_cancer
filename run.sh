#!/bin/bash

# Check if a command line argument (CSV file path) is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path/to/your/testing.csv"
    exit 1
fi

CSV_FILE_PATH=$1

# Run the Python script with the CSV file as an argument
python -m inference $CSV_FILE_PATH
