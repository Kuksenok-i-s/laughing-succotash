#!/usr/bin/env python3
"""Pull a restricted archive from Xavier; verify before retaining or rotating it."""
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
from assistant_backup import restore


def check_freshness(path, now=None):
    with tarfile.open(path) as archive:
        timestamp = archive.getmember('data/core.sqlite3').mtime
    age = (time.time() if now is None else now) - timestamp
    if age > 36*3600 or age < -3600:
        raise ValueError('Source backup is stale or has an invalid timestamp')


def main():
    os.umask(0o077)
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
                    '-o','ServerAliveCountMax=3', 'assistant-backup@10.0.7.127'],
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
            print(json.dumps(report),flush=True)
        finally:
            partial.unlink(missing_ok=True)
            if target.exists():
                shutil.rmtree(target)


if __name__ == '__main__':
    main()
