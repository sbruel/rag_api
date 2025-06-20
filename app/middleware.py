# app/middleware.py
import os
import jwt
import uuid
from jwt import PyJWTError
from fastapi import Request
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from contextvars import ContextVar
from app.config import logger

# Context variable to store the trace ID
trace_id_var: ContextVar[str] = ContextVar('trace_id', default=None)


async def correlation_middleware(request: Request, call_next):
    """Add OpenTelemetry-compatible trace ID to each request for log tracing"""
    # Generate or extract trace ID following OpenTelemetry standard
    # Check for OpenTelemetry traceparent header first, then fallback to custom headers
    traceparent = request.headers.get("traceparent")
    if traceparent:
        # Extract trace ID from traceparent header (format: 00-<trace_id>-<span_id>-<flags>)
        try:
            trace_id = traceparent.split("-")[1]
        except (IndexError, ValueError):
            trace_id = None
    else:
        # Check for custom trace ID headers
        trace_id = (request.headers.get("X-Trace-ID") or 
                   request.headers.get("X-Request-ID") or 
                   request.headers.get("X-Correlation-ID"))
    
    # Generate new trace ID if none provided (32-char hex string)
    if not trace_id:
        trace_id = uuid.uuid4().hex + uuid.uuid4().hex[:16]  # 32 characters
    
    # Store in context variable
    trace_id_var.set(trace_id)
    
    # Store in request state for access in routes
    request.state.trace_id = trace_id
    
    response = await call_next(request)
    
    # Add trace ID to response headers (OpenTelemetry standard)
    response.headers["X-Trace-ID"] = trace_id
    # Also add as X-Request-ID for compatibility
    response.headers["X-Request-ID"] = trace_id
    
    return response


async def security_middleware(request: Request, call_next):
    async def next_middleware_call():
        return await call_next(request)

    if request.url.path in {"/docs", "/openapi.json", "/health"}:
        return await next_middleware_call()

    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        logger.warn("JWT_SECRET not found in environment variables")
        return await next_middleware_call()

    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        logger.info(
            f"Unauthorized request with missing or invalid Authorization header to: {request.url.path}"
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid Authorization header"},
        )

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        exp_timestamp = payload.get("exp")
        if exp_timestamp and datetime.now(tz=timezone.utc) > datetime.fromtimestamp(
            exp_timestamp, tz=timezone.utc
        ):
            logger.info(
                f"Unauthorized request with expired token to: {request.url.path}"
            )
            return JSONResponse(
                status_code=401, content={"detail": "Token has expired"}
            )

        request.state.user = payload
        logger.debug(f"{request.url.path} - {payload}")
    except PyJWTError as e:
        logger.info(
            f"Unauthorized request with invalid token to: {request.url.path}, reason: {str(e)}"
        )
        return JSONResponse(
            status_code=401, content={"detail": f"Invalid token: {str(e)}"}
        )

    return await next_middleware_call()