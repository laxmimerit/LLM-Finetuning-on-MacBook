"""Check a predicted SQL query by running it.

sql-create-context has table schemas but no rows, so for each question we build
3 small SQLite databases with random rows (seeded with the literal values from the
reference query, so WHERE clauses match something). A prediction is correct only
if it returns the same rows as the reference query on all 3 databases.
"""
import random
import re
import sqlite3
from collections import Counter

LITERAL = re.compile(r'(\w+)\s*(?:=|>|<|>=|<=|LIKE)\s*("[^"]*"|\'[^\']*\'|-?\d+(?:\.\d+)?)', re.I)
WORDS = ["alpha", "beta", "gamma", "delta", "india", "june 1", "w", "l", "2008", "none"]


def extract_sql(raw):
    """Pull the SQL out of a code block or surrounding text."""
    m = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", raw, re.S | re.I)
    s = m.group(1) if m else raw
    m = re.search(r"(SELECT\b.*?)(;|\n\n|$)", s, re.S | re.I)
    return m.group(1).strip() if m else s.strip()


def build_db(ex, seed):
    rng = random.Random(seed)
    con = sqlite3.connect(":memory:")
    # The dataset lowercases many values ("june 22") while questions don't: compare text case-insensitively.
    schema = re.sub(r"\bVARCHAR\b", "TEXT COLLATE NOCASE", ex["context"], flags=re.I)
    for stmt in schema.split(";"):
        if stmt.strip():
            con.execute(stmt)
    hints = {}
    for col, lit in LITERAL.findall(ex["answer"]):
        lit = lit.strip("\"'")
        pool = hints.setdefault(col.lower(), [])
        pool.append(lit)
        try:   # numbers: also add neighbours so >, < conditions split the rows
            n = float(lit)
            pool += [str(int(n) - 1), str(int(n) + 1)]
        except ValueError:
            pass
    for table, cols in re.findall(r"CREATE TABLE (\w+) \((.*?)\)", ex["context"], re.I):
        names = [c.strip().split()[0] for c in cols.split(",")]
        for _ in range(25):
            row = []
            for c in names:
                pool = hints.get(c.lower())
                if pool and rng.random() < 0.6:
                    row.append(rng.choice(pool))
                elif rng.random() < 0.5:
                    row.append(rng.randint(0, 30))
                else:
                    row.append(rng.choice(WORDS))
            con.execute(f"INSERT INTO {table} VALUES ({','.join('?' * len(row))})", row)
    return con


def run(con, sql):
    rows = con.execute(sql).fetchall()
    return Counter(tuple(str(v).lower() for v in r) for r in rows)   # order doesn't matter


def check(ex, sql):
    """'correct', 'wrong', or 'error' (the query doesn't run)."""
    for seed in range(3):
        con = build_db(ex, seed)
        gold = run(con, ex["answer"])
        try:
            pred = run(con, sql)
        except Exception:
            return "error"
        if pred != gold:
            return "wrong"
    return "correct"
