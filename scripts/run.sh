#!/bin/bash
# Load environment variables from .env file directly
if [ -f ".env" ]; then
    # Read .env line by line and export each variable
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ $line =~ ^#.* ]] || [ -z "$line" ] && continue
        # Export the variable
        export "$line"
    done < .env
else
    echo "Error: .env file not found."
    exit 1
fi

# Verify required environment variables
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GOOGLE_API_KEY is not set in .env file."
    exit 1
fi

# Set Python path
export PYTHONPATH=$PYTHONPATH:$PWD

# Run the script
python multimodal_search/main.py \
    --gallery_path "./data/collections_test" \
    --collection_name "collections_test" \
    --query "dogs"