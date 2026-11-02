import asyncio
import json

from websockets.asyncio.client import connect


class ServerConnection:
    def __init__(self, server_url: str, retries: int = 3, delay: float = 0.5):
        self.server_url = server_url
        self.websocket = None
        self.retries = retries
        self.delay = delay
        self.lock = asyncio.Lock()

    async def connect(self):
        self.websocket = await connect(self.server_url, open_timeout=2, close_timeout=1)
        return self.websocket

    async def close(self):
        socket, self.websocket = self.websocket, None
        if socket is not None:
            await socket.close()

    async def request(self, metadata: dict, image: bytes | None = None):
        # Retry the complete JSON/JPEG/response transaction, never an individual message.
        async with self.lock:
            for attempt in range(self.retries):
                try:
                    if self.websocket is None:
                        await self.connect()
                    await self.websocket.send(json.dumps(metadata))
                    if image is not None:
                        await self.websocket.send(image)
                    response = json.loads(await self.websocket.recv())
                    if response.get("frame_id") != metadata["frame_id"]:
                        raise ValueError("Recognition frame_id mismatch")
                    return response
                except asyncio.CancelledError:
                    await self.close()
                    raise
                except Exception:
                    await self.close()
                    if attempt + 1 == self.retries:
                        raise
                    await asyncio.sleep(self.delay)
