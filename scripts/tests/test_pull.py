import importlib.util
from pathlib import Path
import sys
import tarfile
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
spec = importlib.util.spec_from_file_location('pull',Path(__file__).parents[1]/'assistant_backup_pull.py')
pull = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pull)


@pytest.mark.parametrize('timestamp,ok', [(100000, True),(1,False),(200000,False)])
def test_source_archive_freshness(tmp_path,timestamp,ok):
    archive=tmp_path/'backup.tar'
    with tarfile.open(archive,'w') as dst:
        info=tarfile.TarInfo('data/core.sqlite3')
        info.mtime=timestamp
        dst.addfile(info)
    if ok:
        pull.check_freshness(archive,now=150000)
    else:
        with pytest.raises(ValueError):
            pull.check_freshness(archive,now=150000)
