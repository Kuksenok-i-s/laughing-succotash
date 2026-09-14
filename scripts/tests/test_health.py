import importlib.util
from pathlib import Path
from datetime import datetime,timezone
import sqlite3

spec = importlib.util.spec_from_file_location('health', Path(__file__).parents[1]/'assistant_health.py')
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


def test_stale_queue_and_delivery_are_reported_without_payloads(tmp_path):
    path = tmp_path/'core.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE jobs(status TEXT,created_at TEXT,started_at TEXT,finished_at TEXT)')
        db.execute('CREATE TABLE outbound_events(status TEXT,created_at TEXT)')
        db.execute("INSERT INTO jobs VALUES ('queued','2026-09-05T00:00:00+00:00',NULL,NULL)")
        db.execute("INSERT INTO outbound_events VALUES ('pending','2026-09-05T00:00:00+00:00')")
    report = health.queue_report(path,datetime(2026,9,5,1,tzinfo=timezone.utc))
    assert report['oldest_queued_seconds']==3600
    assert len(report['warnings'])==2
    assert report['last_completed_at'] is None
    with sqlite3.connect(path) as db:
        db.execute('DELETE FROM jobs');db.execute('DELETE FROM outbound_events')
    assert health.queue_report(path)['warnings']==[]
