"""Incomplete generations and transient rename failures cannot replace a release."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import dataset_artifacts as a

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    released = root / 'generation'
    pending = root / '.generation.partial'
    pending.mkdir()
    a.write_json(pending / 'manifest.json', {'complete':True})
    assert not released.exists()
    a.publish_directory(pending, released)
    original = a.sha256(released / 'manifest.json')
    pending.mkdir()
    a.write_json(pending / 'manifest.json', {'complete':False})
    try:
        a.publish_directory(pending, released)
    except FileExistsError:
        pass
    else:
        raise AssertionError('An existing frozen generation was overwritten')
    assert a.sha256(released / 'manifest.json') == original
    retry_pending = root / '.retry.partial'
    retry_pending.mkdir()
    a.write_json(retry_pending / 'manifest.json', {'complete':True})
    rename = Path.rename
    calls = []
    def locked_once(path, target):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError('temporary Windows reader lock')
        return rename(path,target)
    with patch.object(Path,'rename',locked_once),patch.object(a.time,'sleep'):
        a.publish_directory(retry_pending,root/'retry')
    assert len(calls)==2 and (root/'retry'/'manifest.json').is_file()
    destination = root / 'pointer.json'
    a.write_json(destination, {'generation':'old'})
    original = a.sha256(destination)
    with patch.object(a.os, 'replace', side_effect=PermissionError('reader lock')), patch.object(a.time, 'sleep'):
        try:
            a.write_json(destination, {'generation':'new'})
        except PermissionError:
            pass
        else:
            raise AssertionError('Expected publication failure')
    assert a.sha256(destination) == original, 'failed publication must preserve old pointer'
    assert list(root.glob('.*.partial')), 'failed new contents must remain recoverable'
print('ok artifact publication: hidden partials, immutable releases, prior pointer preserved on failure')
