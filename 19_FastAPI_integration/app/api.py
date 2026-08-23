# lesson20/app/api.py
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.service import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


# =====================================================================
# Request / Response Schemas
# =====================================================================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user query.")
    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session or conversation UUID.",
    )


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    thread_id: str


# =====================================================================
# Dependencies
# =====================================================================
class AuthenticatedUser(BaseModel):
    user_id: str
    tenant_id: str


async def get_current_user(
    x_user_id: Optional[str] = Header("usr_dev_001"),
    x_tenant_id: Optional[str] = Header("tenant_alpha"),
) -> AuthenticatedUser:
    """Extracts and validates security claims from request headers."""
    if not x_user_id or not x_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required authentication headers.",
        )
    return AuthenticatedUser(user_id=x_user_id, tenant_id=x_tenant_id)


def get_chat_service(request: Request) -> ChatService:
    """Retrieves the ChatService singleton instantiated during lifespan."""
    return request.app.state.chat_service


# =====================================================================
# Route Handlers
# =====================================================================
@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    """Synchronous chat endpoint for standard request/response cycles."""
    try:
        result = await service.chat(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            conversation_id=request.conversation_id,
            message=request.message,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
):
    """Server-Sent Events (SSE) streaming endpoint."""
    return StreamingResponse(
        service.stream_chat(
            tenant_id=user.tenant_id,
            user_id=user.user_id,
            conversation_id=request.conversation_id,
            message=request.message,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables proxy buffering in Nginx
        },
    )