from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

MODEL_NAME = "gemma4:26b"

# Initialize base chat model
model = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


# 1. Define the Pydantic data contract
class TicketAnalysis(BaseModel):
    category: Literal[
        "authentication",
        "billing",
        "technical",
        "account",
        "other",
    ] = Field(description="The primary domain/category of the support ticket.")
    
    priority: Literal[
        "low", 
        "medium", 
        "high",
    ] = Field(description="Calculated urgency: low, medium, or high.")
    
    summary: str = Field(
        description="A concise one-sentence summary of the core issue."
    )
    
    requires_human: bool = Field(
        description="Whether this ticket requires immediate escalation to a human agent."
    )


def exercise_1_structured_output():
    print("=== Exercise 1: Structured Output with Pydantic ===")
    
    # Declarative wrapper producing a Runnable that enforces TicketAnalysis
    structured_model = model.with_structured_output(TicketAnalysis)

    tickets = [
        "I cannot log into my account after changing my password.",
        "I was charged twice for the same subscription this month.",
        "The API is returning HTTP 500 internal server errors in production across all endpoints.",
    ]

    for i, ticket in enumerate(tickets, 1):
        print(f"\n--- Processing Ticket {i} ---")
        print(f"Input Text: {ticket}")
        
        result: TicketAnalysis = structured_model.invoke(ticket)
        
        print(f"Returned Type: {type(result)}")
        print(f"  Category:       {result.category}")
        print(f"  Priority:       {result.priority}")
        print(f"  Requires Human: {result.requires_human}")
        print(f"  Summary:        {result.summary}")


def exercise_2_include_raw():
    print("\n=== Exercise 2: include_raw=True Debug Inspection ===")
    
    raw_structured_model = model.with_structured_output(
        TicketAnalysis, 
        include_raw=True,
    )
    
    ticket = "My billing address updated, but my card was declined."
    result = raw_structured_model.invoke(ticket)
    
    print(f"Raw Output Type:          {type(result['raw'])}")
    print(f"Parsed Object Type:      {type(result['parsed'])}")
    print(f"Parsing Error:           {result['parsing_error']}")
    print(f"Raw Message Payload:     {result['raw'].content}")
    print(f"Validated Model Object:  {result['parsed']}")


def exercise_3_strict_validation_failure():
    print("\n=== Exercise 3: Validation Boundary / Schema Constraint Test ===")

    # Define an artificially constrained schema missing 'high'
    class ConstrainedTicket(BaseModel):
        category: Literal["billing", "account", "technical"]
        priority: Literal["low", "medium"] = Field(
            description="Urgency level. NOTE: High is strictly forbidden; assign medium if critical."
        )
        summary: str

    constrained_model = model.with_structured_output(
        ConstrainedTicket, 
        include_raw=True,
    )

    critical_ticket = "URGENT CRITICAL OUTAGE: The entire database cluster is down!"
    result = constrained_model.invoke(critical_ticket)

    print(f"Raw model response: {result['raw'].content}")
    if result["parsing_error"]:
        print(f"Validation Failure Caught: {result['parsing_error']}")
    else:
        print(f"Model adhered to constraints: {result['parsed']}")


if __name__ == "__main__":
    exercise_1_structured_output()
    exercise_2_include_raw()
    exercise_3_strict_validation_failure()