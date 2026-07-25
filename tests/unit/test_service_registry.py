import pytest

from capsule_brain.runtime.registry import ServiceRegistry
from capsule_brain.runtime.service import CapsuleService, ServiceState


class ServiceA(CapsuleService):
    name = "a"


class ServiceB(CapsuleService):
    name = "b"


@pytest.mark.asyncio
async def test_dependency_order_and_reverse_shutdown():
    events = []

    class A(ServiceA):
        async def start(self):
            events.append("start-a")
            self.state = ServiceState.RUNNING

        async def stop(self):
            events.append("stop-a")
            self.state = ServiceState.STOPPED

    class B(ServiceB):
        async def start(self):
            events.append("start-b")
            self.state = ServiceState.RUNNING

        async def stop(self):
            events.append("stop-b")
            self.state = ServiceState.STOPPED

    registry = ServiceRegistry()
    registry.register(B(), requires=["a"])
    registry.register(A())

    await registry.start_all()
    await registry.stop_all()

    assert events == ["start-a", "start-b", "stop-b", "stop-a"]


def test_missing_dependency_rejected():
    registry = ServiceRegistry()
    registry.register(ServiceB(), requires=["a"])

    with pytest.raises(RuntimeError):
        registry._resolve_start_order()
