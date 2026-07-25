import asyncio,sys,pytest
from capsule_brain.execution.models import ExecutionPolicy,ExecutionRequest
from capsule_brain.execution.runner import ExecutionRunner
@pytest.mark.asyncio
async def test_python_runs(tmp_path):
    root=tmp_path/"sandbox";root.mkdir();(root/"ok.py").write_text("print('ok')")
    runner=ExecutionRunner(ExecutionPolicy(allow=True,cwd_root=str(root),allowed_commands=(sys.executable,),timeout_s=5))
    # validate_request compares basename, so use interpreter basename in policy
    runner.policy.allowed_commands=(__import__("pathlib").Path(sys.executable).name,)
    r=await asyncio.wait_for(runner.run(ExecutionRequest(command=[sys.executable,"ok.py"],cwd=str(root))),timeout=8)
    assert r.passed and "ok" in r.stdout
