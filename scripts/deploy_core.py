#!/usr/bin/env python3
"""Apply an explicitly enumerated Core patch, restoring originals if verification fails.

Run as root on Xavier. Does not install dependencies, run migrations or replace config/secrets.
Manifest paths are relative to the installed agent-core directory and must be .py files; before=null explicitly permits a new file.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_sha(path):
    return sha(path) if path.exists() else None


def busy(database):
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        return db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]


def run(*args):
    return subprocess.check_output(args, text=True, timeout=40).strip()


def health():
    invocation = run('systemctl', 'show', 'agent-core', '-p', 'InvocationID', '--value')
    for _ in range(30):
        if run('systemctl', 'is-active', 'agent-core') != 'active':
            raise RuntimeError('Core is not active')
        if run('systemctl', 'show', 'agent-core', '-p', 'InvocationID', '--value') != invocation:
            raise RuntimeError('Core restarted during verification')
        logs = run('journalctl', '_SYSTEMD_INVOCATION_ID='+invocation, '--no-pager', '-o', 'cat')
        if 'handshake complete with gateway' in logs:
            return
        time.sleep(2)
    raise RuntimeError('Core did not reconnect to Gateway')


def apply(root, stage, backup, manifest, stop, start, verify, idle):
    root, stage = root.resolve(), stage.resolve()
    entries = manifest['files']
    if not entries:
        raise ValueError('Empty manifest')
    checked = []
    for name, hashes in entries.items():
        relative = Path(name)
        destination, candidate = root / relative, stage / relative
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.py':
            raise ValueError('Invalid patch path')
        if destination.resolve() != destination or candidate.resolve() != candidate:
            raise ValueError('Symlinks are forbidden in patch paths')
        if installed_sha(destination) != hashes['before'] or sha(candidate) != hashes['after']:
            raise ValueError('File differs from reviewed manifest: '+name)
        compile(candidate.read_bytes(), name, 'exec')
        checked.append((name, destination, candidate))
    if not idle():
        raise RuntimeError('Core has queued or running jobs; retry when idle')
    backup.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, destination, _ in checked:
        saved = backup / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.copy2(destination, saved)
    (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    stopped = False
    try:
        stop()
        stopped = True
        for name, destination, candidate in checked:
            if installed_sha(destination) != entries[name]['before']:
                raise RuntimeError('Installed file changed during preparation')
            temporary = destination.with_name(destination.name+'.deploy-new')
            try:
                shutil.copy2(candidate, temporary)
                stat = destination.stat() if destination.exists() else destination.parent.stat()
                os.chown(temporary, stat.st_uid, stat.st_gid)
                os.chmod(temporary, (stat.st_mode & 0o777) if destination.exists() else 0o644)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        start()
        verify()
    except BaseException:
        if stopped:
            stop()
            for name, destination, _ in checked:
                if entries[name]["before"] is None:
                    destination.unlink(missing_ok=True)
                else:
                    shutil.copy2(backup / name, destination)
            start()
            verify()
        raise
    (backup / 'installed-ok').write_text(datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.stage / 'manifest.json').read_text())
    backup = Path('/home/nvidia/core-releases') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    apply(Path('/home/nvidia/tg_bot_kirpich/agent-core'), args.stage, backup, manifest,
          lambda: run('systemctl', 'stop', 'agent-core'),
          lambda: run('systemctl', 'start', 'agent-core'), health,
          lambda: not busy(Path('/home/nvidia/assistant/core.sqlite3')))
    print(json.dumps({'deployed':True, 'rollback_files':str(backup)}))


if __name__ == '__main__':
    main()
