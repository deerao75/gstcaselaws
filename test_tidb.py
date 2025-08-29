# test_tidb.py
import os, ssl, sys, os.path as op
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# load .env if present
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# required vars
user = os.environ["TIDB_USER"]
pwd  = os.environ["TIDB_PASSWORD"]
host = os.environ["TIDB_HOST"]
port = int(os.environ.get("TIDB_PORT", "4000"))
db   = os.environ["TIDB_DB"]

# pick a CA file
ca = os.environ.get("TIDB_SSL_CA", "")
if not ca:
    try:
        import certifi
        ca = certifi.where()
    except Exception:
        ca = ""  # will error below with a clear message

if not ca or not op.exists(ca):
    print("ERROR: CA file not found.\n"
          "Set TIDB_SSL_CA to a valid path (e.g., certifi’s cacert.pem) or download TiDB CA.\n"
          "Tip: python -c \"import certifi; print(certifi.where())\"")
    sys.exit(1)

print("Using CA:", ca)

url = URL.create(
    "mysql+pymysql",
    username=user, password=pwd, host=host, port=port, database=db,
    query={"charset": "utf8mb4"}
)
engine = create_engine(url, connect_args={"ssl": ssl.create_default_context(cafile=ca)}, pool_pre_ping=True)

with engine.connect() as conn:
    print("TiDB version:", conn.execute(text("SELECT tidb_version()")).scalar())
    try:
        print("acer_details rows:", conn.execute(text("SELECT COUNT(*) FROM acer_details")).scalar())
    except Exception as e:
        print("Table check skipped:", e)
