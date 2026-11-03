import sqlite3
from datetime import date, timedelta

import pytest

from dqmonitor.checks import Config, run_checks
from dqmonitor.generate_data import START, N_DAYS, build_orders, write_db

AS_OF = START + timedelta(days=N_DAYS - 1)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "orders.db"
    write_db(build_orders(seed=7), db)
    with sqlite3.connect(db) as con:
        return run_checks(con, AS_OF, START, Config())


def day(n: int) -> str:
    return (START + timedelta(days=n)).isoformat()


def found(alerts, check, d):
    return any(a.check == check and a.day == d for a in alerts)


def test_detects_every_injected_anomaly(result):
    _, alerts = result
    assert found(alerts, "volumen", day(95))      # caída
    assert found(alerts, "volumen", day(70))      # pico por carga duplicada
    assert found(alerts, "duplicados", day(70))
    for d in (100, 101, 102):
        assert found(alerts, "nulos", day(d))
    assert found(alerts, "rango", day(108))
    assert found(alerts, "frescura", AS_OF.isoformat())


def test_no_false_positives(result):
    _, alerts = result
    injected = {day(d) for d in (70, 95, 100, 101, 102, 108)} | {AS_OF.isoformat()}
    assert {a.day for a in alerts} <= injected


def test_clean_table_has_no_alerts():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE orders(order_id INT, order_date TEXT, customer_id INT, country TEXT, channel TEXT, amount_eur REAL)")
    rows, oid = [], 1
    for d in range(60):
        dd = (date(2026, 1, 1) + timedelta(days=d)).isoformat()
        for _ in range(100):
            rows.append((oid, dd, oid, "ES", "web", 20.0)); oid += 1
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", rows)
    _, alerts = run_checks(con, date(2026, 3, 1), date(2026, 1, 1))
    assert alerts == []
