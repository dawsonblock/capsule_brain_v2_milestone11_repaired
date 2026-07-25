import pytest

from capsule_brain.runtime.application import CapsuleApplication
from capsule_brain.runtime.service import ServiceState


@pytest.mark.asyncio
async def test_application_core_lifecycle():
    app = CapsuleApplication()
    app.install_core_services()

    await app.start()

    bus = app.services.require("event_bus")
    assert bus.state == ServiceState.RUNNING

    app.request_shutdown("test")
    await app.wait()
    assert app.shutdown_reason == "test"

    await app.stop()
    assert bus.state == ServiceState.STOPPED
