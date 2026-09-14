#!/usr/bin/env python3
"""Read-only operational report: queue ages, last completion, endpoints and service state."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import urllib.request


def queue_report(database, now=None):
    now = now or datetime.now(timezone.utc)
    warnings = []
    def age(value):
        if not value:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0, int((now-parsed).total_seconds()))
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        counts = dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status'))
        queued = age(db.execute("SELECT min(created_at) FROM jobs WHERE status='queued'").fetchone()[0])
        running = age(db.execute("SELECT min(started_at) FROM jobs WHERE status='running'").fetchone()[0])
        last_success = db.execute("SELECT max(finished_at) FROM jobs WHERE status='completed'").fetchone()[0]
        pending, oldest = db.execute("SELECT count(*),min(created_at) FROM outbound_events WHERE status='pending'").fetchone()
        pending_age = age(oldest)
    if queued is not None and queued > 900:
        warnings.append('queued_job_older_than_15_minutes')
    if running is not None and running > 7200:
        warnings.append('running_job_older_than_2_hours')
    if pending_age is not None and pending_age > 300:
        warnings.append('outbound_delivery_older_than_5_minutes')
    return {'jobs':counts, 'oldest_queued_seconds':queued, 'oldest_running_seconds':running,
            'last_completed_at':last_success, 'pending_deliveries':pending,
            'oldest_delivery_seconds':pending_age, 'warnings':warnings}


def endpoint(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        data = json.load(response)
    if data.get('status') != 'ok' or data.get('core_connected') is False:
        raise ValueError('Endpoint reports unhealthy state')
    return {k:v for k,v in data.items() if k in ('status','core_connected','model_loaded','queued','model')}


def service(name):
    text = subprocess.check_output(['systemctl','show',name,'-p','ActiveState','-p','SubState',
        '-p','NRestarts','-p','MemoryCurrent','-p','MemoryMax'], text=True, timeout=5)
    values = dict(line.split('=',1) for line in text.splitlines() if '=' in line)
    if values.get('ActiveState') != 'active':
        raise ValueError('Service is not active: '+name)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--endpoint', action='append', default=[], help='NAME=URL')
    parser.add_argument('--service', action='append', default=[])
    parser.add_argument('--backup-directory', type=Path)
    parser.add_argument('--require-verified-backup', action='store_true')
    args = parser.parse_args()
    report = {'at':datetime.now(timezone.utc).isoformat(), 'checks':{}, 'errors':[]}
    checks = {}
    if args.database:
        checks['core_queues'] = lambda:queue_report(args.database)
    for name in args.service:
        checks['service:'+name] = lambda name=name:service(name)
    for item in args.endpoint:
        name,url = item.split('=',1)
        checks['endpoint:'+name] = lambda url=url:endpoint(url)
    if args.backup_directory:
        def backup():
            archives = [p for p in args.backup_directory.glob('*.tar') if not p.is_symlink()]
            if not archives:
                raise ValueError('No backup archives found')
            latest = max(archives,key=lambda p:p.stat().st_mtime)
            seconds = datetime.now(timezone.utc).timestamp()-latest.stat().st_mtime
            if seconds > 36*3600:
                raise ValueError('Latest local archive older than 36 hours')
            verified = False
            report_path = args.backup_directory/'last-success.json'
            if report_path.exists():
                proof = json.loads(report_path.read_text())
                verified = (proof.get('archive') == latest.name
                            and proof.get('off_host_verified') is True
                            and proof.get('checksums') == 'ok'
                            and proof.get('sqlite_integrity') == 'ok')
            if args.require_verified_backup and not verified:
                raise ValueError('No successful off-host restore verification for latest archive')
            return {'archive':latest.name, 'age_seconds':int(seconds), 'off_host_verified':verified}
        checks['local_backup'] = backup
    for name,check in checks.items():
        try:
            result = check()
            report['checks'][name] = result
            report['errors'].extend(name+': '+w for w in result.get('warnings',[]))
        except Exception as exc:
            report['errors'].append(name+': '+type(exc).__name__+': '+str(exc))
    report['ok'] = bool(checks) and not report['errors']
    print(json.dumps(report,indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
