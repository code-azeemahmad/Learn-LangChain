import httpx

BASE_URL = "http://127.0.0.1:8000/api/v1/chat"

client = httpx.Client(timeout=30.0)


def send_message(user_id: str, tenant_id: str, conv_id: str, message: str) -> dict:
    headers = {
        "x-user-id": user_id,
        "x-tenant-id": tenant_id,
        "Content-Type": "application/json",
    }
    payload = {
        "message": message,
        "conversation_id": conv_id,
    }
    response = client.post(BASE_URL, json=payload, headers=headers)
    return response.json()


def run_isolation_tests():
    print("=== Step 1: Alice establishes a secret in Conversation 1 ===")
    r1 = send_message(
        user_id="usr_alice",
        tenant_id="tenant_alpha",
        conv_id="conv_100",
        message="Remember that my secret project code is 'PHOENIX-99'.",
    )
    print(f"Thread ID:  {r1['thread_id']}")
    print(f"Response:   {r1['response']}\n")

    print("=== Step 2: Alice recalls the secret in the SAME conversation (conv_100) ===")
    r2 = send_message(
        user_id="usr_alice",
        tenant_id="tenant_alpha",
        conv_id="conv_100",
        message="What is my secret project code?",
    )
    print(f"Thread ID:  {r2['thread_id']}")
    print(f"Response:   {r2['response']}")
    assert "PHOENIX-99" in r2["response"], "Failed: Alice should recall her secret."
    print(" Result: State preserved successfully within the same thread.\n")

    print("=== Step 3: Alice asks in a DIFFERENT conversation (conv_200) ===")
    r3 = send_message(
        user_id="usr_alice",
        tenant_id="tenant_alpha",
        conv_id="conv_200",
        message="What is my secret project code?",
    )
    print(f"Thread ID:  {r3['thread_id']}")
    print(f"Response:   {r3['response']}")
    assert "PHOENIX-99" not in r3["response"], "Failed: Leak across conversations."
    print(" Result: Conversation-level isolation verified.\n")

    print("=== Step 4: Bob sends conv_100 under a DIFFERENT tenant (tenant_beta) ===")
    r4 = send_message(
        user_id="usr_bob",
        tenant_id="tenant_beta",
        conv_id="conv_100",  # Same conversation ID string, but different identity
        message="What is my secret project code?",
    )
    print(f"Thread ID:  {r4['thread_id']}")
    print(f"Response:   {r4['response']}")
    assert "PHOENIX-99" not in r4["response"], "Failed: Leak across tenants/users."
    print(" Result: Tenant & User identity isolation verified.\n")


if __name__ == "__main__":
    run_isolation_tests()