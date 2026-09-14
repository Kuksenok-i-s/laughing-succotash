#!/usr/bin/env python3
"""Daily snapshot; offline copies must be verified before rotation."""
import argparse
from datetime import datetime, timezone
import fcntl
import grp
import json
import os
import re
from pathlib import Path
from assistant_backup import create


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keep', type=int, default=2)
    parser.add_argument('--reader-group')
    args = parser.parse_args()
    if args.keep < 1:
        parser.error('--keep must be positive')
    os.umask(0o077)
    root = args.output.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = create(args.source, root)
        if args.reader_group:
            gid = grp.getgrnam(args.reader_group).gr_gid
            os.chown(path, 0, gid)
            os.chmod(path, 0o640)
            os.chown(root, 0, gid)
            os.chmod(root, 0o750)
        new = root / 'latest.new'
        new.unlink(missing_ok=True)
        new.symlink_to(path.name)
        new.replace(root / 'latest.tar')
        archives = sorted((p for p in root.glob('*.tar') if not p.is_symlink() and re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}\.tar', p.name)), key=lambda p: (p == path, p.stat().st_mtime_ns), reverse=True)
        for old in archives[args.keep:]:
            old.unlink()
        report = {'at': datetime.now(timezone.utc).isoformat(), 'archive':path.name,
                  'restore_verified':False}
        report_path = root / 'last-success.new'
        report_path.write_text(json.dumps(report)+'\n')
        report_path.replace(root / 'last-success.json')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
