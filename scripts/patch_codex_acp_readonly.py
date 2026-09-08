"""Make codex-acp 1.10.0's misleading read-only preset actually read-only.

Upstream uses workspaceWrite in this mode. The bot relies on MCP for mutations,
so preserve approvals but remove built-in filesystem write permission. Run after
installing the pinned adapter; fail closed on any upstream shape/version change.
"""
import json
from pathlib import Path
import sys


def patch(package: Path) -> None:
    assert json.loads((package / 'package.json').read_text())['version'] == '1.10.0'
    path = package / 'dist/index.js'
    source = path.read_text()
    start = source.index('  static ReadOnly = new _AgentMode(')
    end = source.index('  static Agent = new _AgentMode(', start)
    block = source[start:end]
    if 'type: "readOnly"' in block and '"workspace-write"' not in block:
        return
    assert block.count('type: "workspaceWrite"') == 1
    assert block.count('"workspace-write"') == 1
    fixed = block.replace('type: "workspaceWrite"', 'type: "readOnly"')
    fixed = fixed.replace('      writableRoots: [],\n', '')
    fixed = fixed.replace('      excludeTmpdirEnvVar: false,\n', '')
    fixed = fixed.replace('      excludeSlashTmp: false\n', '')
    fixed = fixed.replace('"workspace-write"', '"read-only"')
    backup = path.with_suffix('.js.before-bot-readonly')
    assert not backup.exists(), 'Unexpected partial patch; inspect backup first'
    backup.write_text(source)
    path.write_text(source[:start] + fixed + source[end:])


if __name__ == '__main__':
    patch(Path(sys.argv[1]))
