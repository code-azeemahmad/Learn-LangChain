# lesson20/app/service.py
import json
from typing import AsyncGenerator  # noqa: UP035

from langchain_core.messages import AIMessageChunk, HumanMessage


class ChatService:
    def __init__(self, agent_graph):
        self.agent = agent_graph

    def _derive_thread_id(self, tenant_id: str, user_id: str, conversation_id: str) -> str:
        """Computes the isolated internal thread identifier."""
        return f"tenant_{tenant_id}_user_{user_id}_conv_{conversation_id}"

    async def chat(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message: str,
    ) -> dict:
        """Synchronous chat execution returning final answer."""
        thread_id = self._derive_thread_id(tenant_id, user_id, conversation_id)
        config = {"configurable": {"thread_id": thread_id}}

        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        final_answer = result["messages"][-1].content
        return {
            "conversation_id": conversation_id,
            "response": final_answer,
            "thread_id": thread_id,
        }

    async def stream_chat(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message: str,
    ) -> AsyncGenerator[str, None]:
        """Translates LangGraph execution into normalized Server-Sent Events (SSE)."""
        thread_id = self._derive_thread_id(tenant_id, user_id, conversation_id)
        config = {"configurable": {"thread_id": thread_id}}

        # 1. Emit stream initialized status
        yield f"data: {json.dumps({'event': 'status', 'data': {'stage': 'agent_started'}})}\n\n"

        try:
            async for chunk, metadata in self.agent.astream(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                stream_mode="messages",
            ):
                # A. Model requested a tool execution
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        if tc.get("name"):
                            yield f"data: {json.dumps({'event': 'tool_start', 'data': {'tool': tc['name']}})}\n\n"

                # B. Model emitted a natural language token
                elif isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield f"data: {json.dumps({'event': 'token', 'data': {'text': chunk.content}})}\n\n"

            # 2. Emit terminal completion event
            yield f"data: {json.dumps({'event': 'done', 'data': {'status': 'completed'}})}\n\n"

        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"