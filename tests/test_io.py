import pandas as pd

from peakpredict.common.io import connect, init_raw_store, read_parquet, write_parquet


def test_raw_store_tables_created(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    init_raw_store(con)
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'raw'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert {"athlete", "event", "performance", "scrape_state"} <= names
    con.close()


def test_init_is_idempotent(tmp_path):
    con = connect(tmp_path / "t.duckdb")
    init_raw_store(con)
    init_raw_store(con)  # must not raise
    con.execute(
        "INSERT INTO raw.athlete (pid, name, sex) VALUES (1, 'A', 1)"
    )
    assert con.execute("SELECT count(*) FROM raw.athlete").fetchone()[0] == 1
    con.close()


def test_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"pid": [1, 2], "score": [0.5, -0.5]})
    path = write_parquet(df, tmp_path / "sub" / "x.parquet")
    assert path.exists()
    back = read_parquet(path)
    pd.testing.assert_frame_equal(df, back)
