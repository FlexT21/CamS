import asyncio
from typing import Any, Callable, TypeVar

from websockets.asyncio.client import connect

T = TypeVar("T")


class ServerConnection:
    def __init__(self, server_url: str, retries: int = 3, delay: float = 2.0):
        self.server_url = server_url
        self.websocket = None
        self.retries = retries
        self.delay = delay

    async def connect(self):
        self.websocket = await connect(self.server_url)
        return self.websocket

    async def close(self):
        if self.websocket:
            await self.websocket.close()

    async def _execution_with_retry(self, operation: Callable[[], T]) -> T:
        for attempt in range(self.retries):
            try:
                self.websocket = await connect(self.server_url)
                return operation()
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.retries - 1:
                    await asyncio.sleep(self.delay)

        raise ConnectionError(
            f"Failed to connect to server after {self.retries} attempts."
        )

    async def send_message(self, message: Any, **kwargs: Any) -> None:
        async def operation() -> None:
            if not self.websocket:
                return None

            return await self.websocket.send(message, **kwargs)

        await self._execution_with_retry(operation)

    async def receive_message(self) -> Any:
        async def operation() -> Any:
            if not self.websocket:
                return ""

            return await self.websocket.recv()

        return await self._execution_with_retry(operation)
