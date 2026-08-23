# lesson19/evaluator.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

JUDGE_MODEL = "gemma4:26b"
judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0)


# =====================================================================
# 1. Pydantic Schemas for Structured LLM Grading
# =====================================================================
class CorrectnessGrade(BaseModel):
    score: float = Field(
        description="Score between 0.0 (completely incorrect) and 1.0 (completely accurate)."
    )
    reason: str = Field(
        description="Clear explanation justifying why the actual answer matches or fails the reference."
    )


class GroundednessGrade(BaseModel):
    is_grounded: bool = Field(
        description="True if all factual claims are directly supported by the context; False if any claim is hallucinated."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="List of specific claims made in the answer that have no backing in the context.",
    )
    reason: str = Field(description="Step-by-step assessment of source attribution.")


# =====================================================================
# 2. LLM-as-Judge Evaluators
# =====================================================================
correctness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert evaluator. Compare the Student Answer against the Reference Answer for the given Question. "
            "Grade semantic correctness from 0.0 to 1.0. Do not penalize phrasing differences if the core facts match.",
        ),
        (
            "human",
            "Question: {question}\n\nReference Answer:\n{reference}\n\nStudent Answer:\n{prediction}",
        ),
    ]
)
correctness_chain = correctness_prompt | judge_llm.with_structured_output(CorrectnessGrade)


groundedness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict hallucination detection judge. Determine whether EVERY factual statement in the Student Answer "
            "is derived exclusively from the provided Context. If the student answers that context is insufficient when appropriate, "
            "mark it as fully grounded.",
        ),
        (
            "human",
            "Context:\n{context}\n\nStudent Answer:\n{prediction}",
        ),
    ]
)
groundedness_chain = groundedness_prompt | judge_llm.with_structured_output(GroundednessGrade)


def evaluate_correctness(question: str, reference: str, prediction: str) -> CorrectnessGrade:
    """Evaluates semantic similarity to ground truth."""
    return correctness_chain.invoke(
        {"question": question, "reference": reference, "prediction": prediction}
    )


def evaluate_groundedness(context: str, prediction: str) -> GroundednessGrade:
    """Evaluates hallucination and adherence to provided retrieval context."""
    return groundedness_chain.invoke(
        {"context": context, "prediction": prediction}
    )


# =====================================================================
# 3. Deterministic Heuristic Evaluator
# =====================================================================
def evaluate_exact_boundary(reference: str, prediction: str) -> float:
    """Fast rule-based check for boundary refusal handling."""
    ref_lower = reference.lower()
    pred_lower = prediction.lower()
    if "not have enough information" in ref_lower:
        return 1.0 if "not have enough information" in pred_lower or "insufficient" in pred_lower else 0.0
    return 1.0 if ref_lower in pred_lower else 0.0