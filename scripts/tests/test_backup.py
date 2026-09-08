import importlib.util
import io
from pathlib import Path
import sqlite3
import tarfile

import pytest

spec = importlib.util.spec_from_file_location('backup', Path(__file__).parents[1] / 'assistant_backup.py')
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def test_restore_includes_committed_wal_and_user_files(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    with sqlite3.connect(source / 'core.sqlite3') as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE notes(body TEXT)')
        db.execute("INSERT INTO notes VALUES ('preserve me')")
        db.commit()
        (source / 'note.md').write_text('user file')
        (source / 'tmp').mkdir()
        (source / 'tmp/transient').write_text('ignore')
        archive = backup.create(source, tmp_path / 'copies')
        restored = tmp_path / 'restored'
        assert backup.restore(archive, restored)['checksums'] == 'ok'
        with sqlite3.connect(restored / 'data/core.sqlite3') as restored_db:
            assert restored_db.execute('SELECT body FROM notes').fetchone() == ('preserve me',)
        assert (restored / 'data/note.md').read_text() == 'user file'
        assert not (restored / 'data/tmp').exists()
        with pytest.raises(FileExistsError):
            backup.restore(archive, restored)


@pytest.mark.parametrize('name,kind', [('../escape', 'file'), ('data/link', 'link')])
def test_unsafe_archive_is_rejected(tmp_path, name, kind):
    path = tmp_path / 'bad.tar'
    with tarfile.open(path, 'w') as archive:
        info = tarfile.TarInfo(name)
        if kind == 'link':
            info.type = tarfile.SYMTYPE
            info.linkname = '/etc/passwd'
        archive.addfile(info, io.BytesIO(b''))
    with pytest.raises(ValueError):
        backup.restore(path, tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()
    assert not (tmp_path / 'escape').exists()


def test_corruption_is_rejected_and_partial_restore_removed(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    with sqlite3.connect(source / 'core.sqlite3') as db:
        db.execute('CREATE TABLE example(value TEXT)')
    (source / 'note.txt').write_text('original')
    archive = backup.create(source, tmp_path / 'copies')
    corrupt = tmp_path / 'corrupt.tar'
    with tarfile.open(archive) as src, tarfile.open(corrupt, 'w') as dst:
        for member in src:
            data = src.extractfile(member).read() if member.isfile() else None
            if member.name == 'data/note.txt':
                data = b'tampered'
                member.size = len(data)
            dst.addfile(member, io.BytesIO(data) if data is not None else None)
    with pytest.raises(ValueError, match='checksum mismatch'):
        backup.restore(corrupt, tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()
