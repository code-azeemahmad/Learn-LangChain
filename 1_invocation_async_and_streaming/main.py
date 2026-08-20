import asyncio

from langchain_ollama import ChatOllama

MODEL_NAME = "gemma4:26b"

# Initialize model abstraction with resilience controls
model = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    timeout=60,
    max_retries=2,
)


def run_sync() -> None:
    prompt = "Explain dependency injection in one sentence."
    response = model.invoke(prompt)

    print(f"Type: {type(response)}")
    print(f"Content: {response.content}\n")


async def run_async() -> None:
    prompt = "Explain async programming in one sentence."
    response = await model.ainvoke(prompt)

    print(f"Type: {type(response)}")
    print(f"Content: {response.content}\n")


async def run_streaming() -> None:
    prompt = "Explain event-driven architecture in two short sentences."

    first_chunk = True
    async for chunk in model.astream(prompt):
        if first_chunk:
            print(f"Chunk Type: {type(chunk)}")
            print("Streamed Output: ", end="", flush=True)
            first_chunk = False
        # In langchain_core, chunk.content holds the raw delta token string
        print(chunk.content, end="", flush=True)
    print("\n")


async def main() -> None:
    run_sync()
    await run_async()
    await run_streaming()


if __name__ == "__main__":
    asyncio.run(main())


'''
invoke()
ainvoke()
stream()
astream()
batch()
astream_events()
'''
# AIMessage vs AIMessageChunk