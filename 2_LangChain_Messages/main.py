from langchain.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gemma4:26b",
    temperature=0,
)


messages = [
    SystemMessage(
        content="You are a concise programming tutor."
    ),
    HumanMessage(
        content="What is dependency injection?"
    ),
]

response = model.invoke(messages)

messages.append(response)

messages.append(
    HumanMessage(
        content="Give me a Python example."
    )
)

response = model.invoke(messages)

print(response.content)


response = model.invoke(messages)

print("TYPE:", type(response))
print("CONTENT:", response.content)
print("RESPONSE METADATA:", response.response_metadata)
print("USAGE:", response.usage_metadata)