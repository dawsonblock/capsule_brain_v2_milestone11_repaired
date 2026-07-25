from __future__ import annotations

import asyncio
import logging
import signal
import sys
from collections.abc import Callable

from PyQt5.QtWidgets import QApplication
from qasync import QEventLoop

from capsule_brain.runtime.bootstrap import build_application

log = logging.getLogger(__name__)


async def run_application(
    qt_app: QApplication,
    *,
    create_window: Callable | None = None,
    config: dict | None = None,
) -> None:
    runtime = build_application(config)
    await runtime.start()

    window = None
    if create_window is not None:
        window = create_window(runtime)
        window.show()

    quit_event = asyncio.Event()

    def request_shutdown(reason: str) -> None:
        log.info("Shutdown requested: %s", reason)
        runtime.request_shutdown(reason)
        quit_event.set()

    qt_app.aboutToQuit.connect(lambda: request_shutdown("qt-quit"))

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig,
                lambda s=sig: request_shutdown(s.name),
            )
        except (NotImplementedError, RuntimeError):
            # Windows/event loops without POSIX signal support.
            pass

    try:
        await quit_event.wait()
    finally:
        await runtime.stop()


def main(
    create_window: Callable | None = None,
    *,
    config: dict | None = None,
) -> None:
    qt_app = QApplication.instance() or QApplication(sys.argv)
    loop = QEventLoop(qt_app)
    asyncio.set_event_loop(loop)

    with loop:
        loop.run_until_complete(
            run_application(
                qt_app,
                create_window=create_window,
                config=config,
            )
        )
