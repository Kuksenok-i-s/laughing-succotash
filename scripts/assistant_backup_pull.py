#!/usr/bin/env python3
"""Pull a restricted archive from the Core host; verify before retaining or rotating it.

BACKUP_PULL_REMOTE must be set (see docs/infra.local.md).
"""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
import uuid
import grp
from assistant_backup import restore


def allow_reader(root, *paths):
    try:
        gid = grp.getgrnam('assistant-backup').gr_gid
    except KeyError:
        return
    os.chown(root, 0, gid)
    os.chmod(root, 0o750)
    for path in paths:
        if path.exists():
            os.chown(path, 0, gid)
            os.chmod(path, 0o640)


def check_freshness(path, now=None):
    with tarfile.open(path) as archive:
        timestamp = archive.getmember('data/core.sqlite3').mtime
    age = (time.time() if now is None else now) - timestamp
    if age > 36*3600 or age < -3600:
        raise ValueError('Source backup is stale or has an invalid timestamp')


def main():
    os.umask(0o077)
    remote = os.environ.get('BACKUP_PULL_REMOTE')
    if not remote:
        raise SystemExit('BACKUP_PULL_REMOTE is not set')
    root = Path('/var/lib/assistant-backups')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    key_root = Path('/var/lib/assistant-backup-client')
    with (root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        partial, target, final = root/(name+'.partial'), root/(name+'.restore'), root/(name+'.tar')
        try:
            with partial.open('xb') as stream:
                subprocess.run(['ssh','-F','/dev/null','-i',str(key_root/'id_ed25519'),
                    '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
                    '-o','UserKnownHostsFile='+str(key_root/'known_hosts'),
                    '-o','ConnectTimeout=15','-o','ServerAliveInterval=30',
                    '-o','ServerAliveCountMax=3', remote],
                    stdout=stream,check=True,timeout=3600)
                stream.flush()
                os.fsync(stream.fileno())
            check_freshness(partial)
            result = restore(partial,target)
            partial.replace(final)
            report = {'at':datetime.now(timezone.utc).isoformat(), 'archive':final.name,
                      'off_host_verified':True, **result}
            fresh = root/'last-success.new'
            fresh.write_text(json.dumps(report)+'\n')
            fresh.replace(root/'last-success.json')
            archives = sorted((p for p in root.glob('*.tar') if not p.is_symlink()
                and re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}\.tar', p.name)), key=lambda p: (p == final, p.stat().st_mtime_ns), reverse=True)
            for old in archives[7:]:
                old.unlink()
            allow_reader(root, final, root/'last-success.json', *archives[:7])
            print(json.dumps(report),flush=True)
        finally:
            partial.unlink(missing_ok=True)
            if target.exists():
                shutil.rmtree(target)


if __name__ == '__main__':
    main()
