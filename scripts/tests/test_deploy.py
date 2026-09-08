import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('deploy', Path(__file__).parents[1] / 'deploy_core.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def setup(tmp_path):
    root, stage = tmp_path/'root', tmp_path/'stage'
    root.mkdir(); stage.mkdir()
    (root/'app.py').write_text('value = 1\n')
    (stage/'app.py').write_text('value = 2\n')
    manifest = {'files':{'app.py':{'before':deploy.sha(root/'app.py'), 'after':deploy.sha(stage/'app.py')}}}
    return root, stage, manifest


def test_failed_health_restores_original_and_checks_it(tmp_path):
    root, stage, manifest = setup(tmp_path)
    checks = []
    def verify():
        checks.append((root/'app.py').read_text())
        if len(checks) == 1:
            raise RuntimeError('health failed')
    with pytest.raises(RuntimeError, match='health failed'):
        deploy.apply(root, stage, tmp_path/'backup', manifest, lambda:None, lambda:None, verify, lambda:True)
    assert checks == ['value = 2\n', 'value = 1\n']
    assert (root/'app.py').read_text() == 'value = 1\n'


def test_busy_core_is_not_stopped(tmp_path):
    root, stage, manifest = setup(tmp_path)
    with pytest.raises(RuntimeError, match='jobs'):
        deploy.apply(root, stage, tmp_path/'backup', manifest, lambda:pytest.fail('stopped'), lambda:None, lambda:None, lambda:False)
    assert not (tmp_path/'backup').exists()


def test_changed_installed_file_is_not_overwritten(tmp_path):
    root, stage, manifest = setup(tmp_path)
    (root/'app.py').write_text('value = 3\n')
    with pytest.raises(ValueError, match='manifest'):
        deploy.apply(root, stage, tmp_path/'backup', manifest, lambda:pytest.fail('stopped'), lambda:None, lambda:None, lambda:True)
    assert (root/'app.py').read_text() == 'value = 3\n'


def test_new_file_is_removed_on_rollback(tmp_path):
    root, stage, manifest = setup(tmp_path)
    (stage/'new.py').write_text('value = 3\n')
    manifest['files']['new.py'] = {'before': None, 'after': deploy.sha(stage/'new.py')}
    checks = []
    def verify():
        checks.append((root/'new.py').exists())
        if len(checks) == 1:
            raise RuntimeError('health failed')
    with pytest.raises(RuntimeError, match='health failed'):
        deploy.apply(root, stage, tmp_path/'backup', manifest, lambda:None, lambda:None, verify, lambda:True)
    assert checks == [True, False]


def test_new_file_is_installed(tmp_path):
    root, stage, manifest = setup(tmp_path)
    (stage/'new.py').write_text('value = 3\n')
    manifest['files']['new.py'] = {'before': None, 'after': deploy.sha(stage/'new.py')}
    deploy.apply(root, stage, tmp_path/'backup', manifest, lambda:None, lambda:None, lambda:None, lambda:True)
    assert (root/'new.py').read_text() == 'value = 3\n'
