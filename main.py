from langchain_ollama import ChatOllama

# Creates a LangChain chat model object.
model = ChatOllama(
    model="gemma4:26b",
    temperature=0,
)


# LangChain's integration handles the provider-specific mechanics.
'''response = model.invoke("Explain what an API is in one sentence.")'''
response = model.invoke(
    [
        ("system", "You are a concise programming tutor."),
        ("human", "Explain Python decorators in one sentence."),
    ]
)
'''
This is important because you can already see that a LangChain model
is designed around messages, not merely raw strings.
'''

print(response)
print(type(response))
print(response.content)
print(response.response_metadata)