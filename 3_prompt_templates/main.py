from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

MODEL_NAME = "gemma4:26b"

# Initialize Model
model = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


def exercise_1_basic_reusable_prompt():
    print("=== Exercise 1: Reusable Prompt Template ===")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise Python tutor. Keep answers under 2 sentences."),
            ("human", "Explain {topic} and give one practical example."),
        ]
    )

    topics = ["dependency injection", "asyncio", "FastAPI dependency injection"]

    for topic in topics:
        print(f"\n--- Topic: {topic} ---")
        # 1. Format prompt into structured messages (ChatPromptValue)
        formatted_messages = prompt.invoke({"topic": topic})
        
        # 2. Invoke model with rendered messages
        response = model.invoke(formatted_messages)
        print(response.content)


def exercise_2_multi_variables():
    print("\n=== Exercise 2: Multiple Variables & Input Inspection ===")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a {role}. Keep it brief."),
            ("human", "Explain {topic} for a {experience} developer."),
        ]
    )

    # Inspect variable contract
    print(f"Required input variables: {prompt.input_variables}")

    formatted_messages = prompt.invoke(
        {
            "role": "systems architect",
            "topic": "horizontal vs vertical scaling",
            "experience": "junior",
        }
    )

    # Inspect rendered message structure
    for msg in formatted_messages.messages:
        print(f"[{type(msg).__name__}]: {msg.content}")

    response = model.invoke(formatted_messages)
    print(f"[AIMessage]: {response.content}")


def exercise_3_messages_placeholder():
    print("\n=== Exercise 3: MessagesPlaceholder & History Integration ===")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise programming tutor."),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{question}"),
        ]
    )

    # Turn 1: Invocation without history (optional placeholder defaults to empty)
    print("\n--- Turn 1 (No History) ---")
    turn_1_messages = prompt.invoke({"question": "What is connection pooling in PostgreSQL?"})
    turn_1_response = model.invoke(turn_1_messages)
    print(f"Assistant: {turn_1_response.content}")

    # Turn 2: Invocation with history supplied
    print("\n--- Turn 2 (With History) ---")
    conversation_history = [
        HumanMessage(content="What is connection pooling in PostgreSQL?"),
        turn_1_response,  # AIMessage from turn 1
    ]

    turn_2_messages = prompt.invoke(
        {
            "history": conversation_history,
            "question": "How does PgBouncer help with this?",
        }
    )

    print("Rendered message chain:")
    for msg in turn_2_messages.messages:
        print(f"  - [{type(msg).__name__}]: {msg.content[:60]}...")

    turn_2_response = model.invoke(turn_2_messages)
    print(f"Assistant: {turn_2_response.content}")


def test_missing_variable_failure():
    print("\n=== Test: Missing Variable Error Handling ===")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "Explain {topic}."),
        ]
    )

    try:
        # Intentionally missing the 'topic' key
        prompt.invoke({})
    except KeyError as e:
        print(f"Successfully caught missing variable KeyError: {e}")


if __name__ == "__main__":
    exercise_1_basic_reusable_prompt()
    exercise_2_multi_variables()
    exercise_3_messages_placeholder()
    test_missing_variable_failure()