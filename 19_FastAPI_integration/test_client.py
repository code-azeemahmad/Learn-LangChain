# test_client.py
import asyncio
import json
import httpx

BASE_URL = "http://127.0.0.1:8000/api/v1/chat"
HEADERS = {
    "x-user-id": "usr_dev_101",
    "x-tenant-id": "tenant_alpha",
    "Content-Type": "application/json",
}

# Configure a generous timeout (120 seconds) for local LLM inference
CLIENT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


async def test_sync_endpoint():
    print("=== 1. Testing Sync Endpoint (POST /api/v1/chat) ===")
    payload = {
        "message": "What does PostgreSQL use to handle high write concurrency?",
        "conversation_id": "conv-test-101",
    }
    # Pass CLIENT_TIMEOUT here
    async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
        res = await client.post(BASE_URL, json=payload, headers=HEADERS)
        data = res.json()
        print(f"Status:     {res.status_code}")
        print(f"Thread ID:  {data['thread_id']}")
        print(f"Response:   {data['response']}\n")


async def test_streaming_endpoint():
    print("=== 2. Testing SSE Streaming Endpoint (POST /api/v1/chat/stream) ===")
    payload = {
        "message": "How does FastAPI handle dependency injection according to internal docs?",
        "conversation_id": "conv-test-102",
    }
    # Pass CLIENT_TIMEOUT here
    async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
        async with client.stream("POST", f"{BASE_URL}/stream", json=payload, headers=HEADERS) as response:
            print(f"Connected [{response.status_code}]. Receiving SSE Events:")
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event_payload = json.loads(line[6:])
                    event_type = event_payload["event"]
                    event_data = event_payload["data"]

                    if event_type == "status":
                        print(f"\n[Status]: {event_data['stage']}")
                    elif event_type == "tool_start":
                        print(f"\n[Tool Executing]: {event_data['tool']}...")
                    elif event_type == "token":
                        print(event_data["text"], end="", flush=True)
                    elif event_type == "done":
                        print("\n\n[Stream Completed]")
                    elif event_type == "error":
                        print(f"\n[Error]: {event_data['message']}")


async def main():
    await test_sync_endpoint()
    await test_streaming_endpoint()


if __name__ == "__main__":
    asyncio.run(main())