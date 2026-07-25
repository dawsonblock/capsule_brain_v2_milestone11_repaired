import pytest
from capsule_brain.events.local_bus import LocalEventBus
from capsule_brain.verification.repository import VerificationRepository
from capsule_brain.verification.service import VerificationService
from capsule_brain.verification.models import VerificationStatus
@pytest.mark.asyncio
async def test_python_failure(tmp_path):
    b=LocalEventBus();await b.start();s=VerificationService(b,repository=VerificationRepository(str(tmp_path/"v.sqlite")));await s.start()
    seen=[];b.subscribe("verification.failed",lambda e:seen.append(e.payload))
    r=await s.verify(subject="def x(:\n pass",source="test",metadata={"content_type":"python"})
    assert r.status==VerificationStatus.FAIL and seen and await s.repository.count()==1
    await s.stop();await b.stop()
@pytest.mark.asyncio
async def test_json_schema_pass(tmp_path):
    b=LocalEventBus();await b.start();s=VerificationService(b,repository=VerificationRepository(str(tmp_path/"v.sqlite")));await s.start()
    r=await s.verify(subject='{"name":"capsule","ok":true}',metadata={"content_type":"json","required_fields":["name","ok"],"expected_contains":["capsule"]})
    assert r.status==VerificationStatus.PASS
    await s.stop();await b.stop()
