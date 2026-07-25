import pytest
from capsule_brain.execution.models import ExecutionPolicy,ExecutionRequest
from capsule_brain.execution.policy import validate_request,ExecutionPolicyError
def test_disabled_policy_blocks(tmp_path):
    with pytest.raises(ExecutionPolicyError):
        validate_request(ExecutionPolicy(allow=False,cwd_root=str(tmp_path)),ExecutionRequest(command=["python","x.py"],cwd=str(tmp_path)))
def test_cwd_escape_and_command_block(tmp_path):
    root=tmp_path/"sandbox";root.mkdir()
    policy=ExecutionPolicy(allow=True,cwd_root=str(root),allowed_commands=("python",))
    with pytest.raises(ExecutionPolicyError):
        validate_request(policy,ExecutionRequest(command=["sh","x"],cwd=str(root)))
    with pytest.raises(ExecutionPolicyError):
        validate_request(policy,ExecutionRequest(command=["python","x.py"],cwd=str(tmp_path)))
