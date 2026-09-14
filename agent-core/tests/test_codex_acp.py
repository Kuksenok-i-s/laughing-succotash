import json
import pytest
from agent_core.agent.codex_acp import CodexACPBackend
from agent_core.agent.base import AgentError


def test_codex_launch_has_no_cursor_subcommand_or_model_flag(tmp_path):
    backend = CodexACPBackend('/bin/codex-acp',default_workspace=tmp_path,model='test-model')
    client = backend._make_client()
    assert client._argv == ['/bin/codex-acp']
    config = json.loads(client._env['CODEX_CONFIG'])
    assert config['model'] == 'test-model'
    assert config['features']['shell_tool'] is False
    assert config['features']['apply_patch_freeform'] is False
    assert client._env['INITIAL_AGENT_MODE'] == 'read-only'
    assert backend.name == 'codex-acp'


async def test_codex_protected_mode_maps_to_adapter_mode(tmp_path):
    calls=[]
    class Client:
        running=True
        async def call(self,method,params,**kwargs):
            calls.append((method,params))
    backend=CodexACPBackend(default_workspace=tmp_path)
    backend._client=Client()
    await backend.set_mode('session','plan')
    assert calls==[('session/set_mode',{'sessionId':'session','modeId':'read-only'})]
    with pytest.raises(AgentError):
        await backend.set_mode('session','agent-full-access')


async def test_codex_never_silently_approves_escalation(tmp_path):
    backend=CodexACPBackend(default_workspace=tmp_path)
    assert await backend._on_permission({'options':[
        {'optionId':'yes','kind':'allow_once'}, {'optionId':'no','kind':'reject_once'}
    ]}) == 'no'
    assert await backend._on_permission({'options':[]}) is None


async def test_only_correlated_assistant_mcp_call_is_allowed(tmp_path):
    backend=CodexACPBackend(default_workspace=tmp_path)
    request={'sessionId':'s','toolCall':{'toolCallId':'t'},'_meta':{'is_mcp_tool_approval':True},
             'options':[{'optionId':'yes','kind':'allow_once'},{'optionId':'no','kind':'reject_once'}]}
    assert await backend._on_permission(request)=='no'
    await backend._on_update('s',{'sessionUpdate':'tool_call','toolCallId':'t',
                                 '_meta':{'is_mcp_tool_call':True},'rawInput':{'server':'other'}})
    assert await backend._on_permission(request)=='no'
    await backend._on_update('s',{'sessionUpdate':'tool_call','toolCallId':'t',
                                 '_meta':{'is_mcp_tool_call':True},'rawInput':{'server':'assistant'}})
    assert await backend._on_permission(request)=='yes'
    assert await backend._on_permission({**request,'sessionId':'another'})=='no'
    assert await backend._on_permission({**request,'_meta':{}})=='no'
