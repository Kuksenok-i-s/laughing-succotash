#!/usr/bin/env python3
"""Create and verify portable data backups; never restore over live data.

SQLite is copied through its online backup API. Ordinary files must remain unchanged while
read; this is not a transaction across SQLite and files. Stop writers for that guarantee.
Temporary files and the live database's WAL/SHM are excluded, everything else is included.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_database(path):
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        if db.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise ValueError('SQLite integrity check failed')
        if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise ValueError('SQLite foreign key check failed')


def create(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError('Backup directory must be outside data directory')
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    final = output / (name + '.tar')
    partial = output / (name + '.partial')
    manifest = {'format': 1, 'files': {}, 'excluded': ['tmp/', 'core.sqlite3-wal', 'core.sqlite3-shm']}
    try:
        with tempfile.TemporaryDirectory(dir=output) as scratch:
            snapshot = Path(scratch) / 'core.sqlite3'
            with sqlite3.connect((source / 'core.sqlite3').as_uri() + '?mode=ro', uri=True) as live:
                with sqlite3.connect(snapshot) as target:
                    live.backup(target)
            check_database(snapshot)
            with partial.open('xb') as stream:
                os.chmod(partial, 0o600)
                with tarfile.open(fileobj=stream, mode='w') as archive:
                    for path in sorted(source.rglob('*')):
                        relative = path.relative_to(source)
                        if relative.parts[0] == 'tmp' or str(relative) in ('core.sqlite3-wal', 'core.sqlite3-shm'):
                            continue
                        if path.is_symlink():
                            raise ValueError('Symlinks require explicit backup policy: ' + str(relative))
                        if path.is_dir():
                            archive.add(path, arcname='data/' + relative.as_posix(), recursive=False)
                            continue
                        if not path.is_file():
                            raise ValueError('Unsupported file: ' + str(relative))
                        actual = snapshot if str(relative) == 'core.sqlite3' else path
                        before = actual.stat()
                        checksum = digest(actual)
                        archive.add(actual, arcname='data/' + relative.as_posix(), recursive=False)
                        after = actual.stat()
                        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                            raise ValueError('File changed during backup: ' + str(relative))
                        manifest['files'][relative.as_posix()] = checksum
                    meta = Path(scratch) / 'manifest.json'
                    meta.write_text(json.dumps(manifest, sort_keys=True))
                    archive.add(meta, arcname='manifest.json', recursive=False)
                stream.flush()
                os.fsync(stream.fileno())
        partial.rename(final)
        return final
    finally:
        partial.unlink(missing_ok=True)


def restore(archive_path, target):
    target = target.resolve()
    target.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        with tarfile.open(archive_path, 'r') as archive:
            seen = set()
            for member in archive:
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or member.name in seen:
                    raise ValueError('Unsafe or duplicate archive path')
                if member.name != 'manifest.json' and (not path.parts or path.parts[0] != 'data'):
                    raise ValueError('Unexpected archive entry')
                seen.add(member.name)
                destination = target / path
                if member.isdir():
                    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, destination.open('xb') as dst:
                        os.chmod(destination, 0o600 | (member.mode & 0o100))
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                else:
                    raise ValueError('Archive links and special files are forbidden')
        manifest = json.loads((target / 'manifest.json').read_text())
        if manifest['format'] != 1:
            raise ValueError('Unsupported manifest format')
        files = {p.relative_to(target / 'data').as_posix(): digest(p)
                 for p in (target / 'data').rglob('*') if p.is_file()}
        if files != manifest['files']:
            raise ValueError('Backup checksum mismatch')
        check_database(target / 'data/core.sqlite3')
        return {'files': len(files), 'sqlite_integrity': 'ok', 'checksums': 'ok'}
    except BaseException:
        shutil.rmtree(target)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create_args = sub.add_parser('create')
    create_args.add_argument('--source', type=Path, required=True)
    create_args.add_argument('--output', type=Path, required=True)
    restore_args = sub.add_parser('restore')
    restore_args.add_argument('--archive', type=Path, required=True)
    restore_args.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'create':
        print(create(args.source, args.output))
    else:
        print(json.dumps(restore(args.archive, args.target)))


if __name__ == '__main__':
    main()
