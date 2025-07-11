## Steps to run the application
1. Create folders `data/raw/` in root directory.
2. keep the documents that you want to ingest to the model in the above folder. 
3. Run the following commands
    ```python3
    python -m app.main ingest # To ingest the data in `data/raw/` to the model
    python -m app.main --question "Ask your questions here" 
    ```