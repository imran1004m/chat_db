import os
import json
from sqlalchemy import text

CACHE_FILE = "schema_cache.json"

def get_schema_info(engine, force_refresh=False):
    db_name = engine.url.database
    if os.path.exists(CACHE_FILE) and not force_refresh:
        with open(CACHE_FILE) as f:
            return json.load(f)

    schema_info = {}
    with engine.connect() as conn:
        tables = conn.execute(text("SHOW TABLES")).fetchall()
        for (table,) in tables:
            columns = conn.execute(text(f"SHOW COLUMNS FROM {table}")).fetchall()
            schema_info[table] = [col[0] for col in columns]

    with open(CACHE_FILE, "w") as f:
        json.dump(schema_info, f)

    return schema_info
