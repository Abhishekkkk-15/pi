"""ESC-key interrupt support for stopping the current agent turn."""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional


class AgentInterrupted(Exception):
    """Raised when the user stops the current agent execution (ESC / Ctrl+C)."""


class InterruptController:
    """Background listener that sets a flag when ESC is pressed."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._listening = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def interrupted(self) -> bool:
        return self._event.is_set()

    def clear(self) -> None:
        self._event.clear()

    def trigger(self) -> None:
        self._event.set()

    def check(self) -> None:
        """Raise AgentInterrupted if interrupt was requested."""
        if self._event.is_set():
            raise AgentInterrupted("Execution stopped by user")

    def start(self, *, clear: bool = True) -> None:
        """Begin listening for ESC. Clears any previous interrupt by default."""
        with self._lock:
            if clear:
                self._event.clear()
            if self._listening:
                return
            self._listening = True
            self._thread = threading.Thread(
                target=self._listen_loop,
                name="esc-interrupt-listener",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop listening for ESC (does not clear the interrupt flag)."""
        with self._lock:
            self._listening = False
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.3)

    def pause(self) -> None:
        """Temporarily stop listening so interactive prompts can read the keyboard."""
        self.stop()

    def resume(self) -> None:
        """Resume listening without clearing an existing interrupt flag."""
        self.start(clear=False)

    def _listen_loop(self) -> None:
        if sys.platform == "win32":
            self._listen_windows()
        else:
            self._listen_unix()

    def _listen_windows(self) -> None:
        import msvcrt

        while self._listening:
            try:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    # ESC is 0x1b; arrow/function keys often start with 0x1b/0xe0
                    if key == b"\x1b":
                        time.sleep(0.03)
                        if not msvcrt.kbhit():
                            self._event.set()
                            break
                        # Drain escape sequence (arrows, etc.)
                        while msvcrt.kbhit():
                            msvcrt.getch()
                    elif key in (b"\x00", b"\xe0") and msvcrt.kbhit():
                        msvcrt.getch()
            except Exception:
                pass
            time.sleep(0.05)

    def _listen_unix(self) -> None:
        import select

        while self._listening:
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not readable:
                    continue
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    # Distinguish bare ESC from CSI sequences
                    more, _, _ = select.select([sys.stdin], [], [], 0.03)
                    if not more:
                        self._event.set()
                        break
                    while select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.read(1)
            except Exception:
                time.sleep(0.05)


# Process-wide controller used by the agent turn
interrupt_controller = InterruptController()
