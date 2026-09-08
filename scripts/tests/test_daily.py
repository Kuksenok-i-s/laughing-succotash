import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parents[1]))
spec=importlib.util.spec_from_file_location('daily',Path(__file__).parents[1]/'assistant_backup_daily.py')
daily=importlib.util.module_from_spec(spec)
spec.loader.exec_module(daily)


def test_rotation_keeps_newest_even_if_random_suffix_sorts_first(tmp_path,monkeypatch):
    old=tmp_path/'20260905T000000Z-ffffffff.tar'
    new=tmp_path/'20260905T000000Z-00000000.tar'
    unrelated=tmp_path/'manual.tar'
    for path in (old,new,unrelated):path.write_bytes(b'archive')
    monkeypatch.setattr(daily,'create',lambda *args:new)
    monkeypatch.setattr(sys,'argv',['daily','--source','unused','--output',str(tmp_path),'--keep','1'])
    daily.main()
    assert new.exists() and unrelated.exists() and not old.exists()
    assert (tmp_path/'latest.tar').resolve()==new
    assert json.loads((tmp_path/'last-success.json').read_text())['archive']==new.name
