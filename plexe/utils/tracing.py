"""
OpenTelemetry tracing decorators for agents and tools.
"""

import inspect
import json
import logging
from collections.abc import Callable
from functools import wraps

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from smolagents import Tool

from plexe.config import Config

logger = logging.getLogger(__name__)

try:
    from openinference.semconv.trace import SpanAttributes

    AGENT_NAME_KEY = SpanAttributes.AGENT_NAME
except Exception:
    AGENT_NAME_KEY = "agent.name"


# Tracer instance for workflow spans
tracer = trace.get_tracer("model_builder_v2.workflow")


def setup_opentelemetry(config: Config):
    """Initialize OpenTelemetry tracing with backend-agnostic configuration."""

    if not config.enable_otel:
        logger.info("OpenTelemetry tracing disabled")
        return

    # Validate endpoint is provided
    if not config.otel_endpoint:
        logger.warning("OpenTelemetry enabled but no endpoint configured - disabling tracing")
        return

    # Configure tracer provider
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)

    # Create OTLP exporter with configured endpoint and headers
    exporter = OTLPSpanExporter(endpoint=config.otel_endpoint, headers=config.otel_headers)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Auto-instrument AI frameworks
    SmolagentsInstrumentor().instrument()  # smolagents

    logger.info(f"OpenTelemetry tracing enabled: {config.otel_endpoint}")


def agent_span(name: str, *, framework: str | None = None, extra: dict | None = None):
    """
    Wrap an agent call in a named span for observability.

    Args:
        name: Span name
        framework: Optional framework tag (e.g., "smolagents")
        extra: Optional additional span attributes

    Example:
        >>> @agent_span("FeatureProcessorAgent")
        >>> def run(self, task: str):
        >>>     # Agent logic
        >>>     pass
    """
    tracer = trace.get_tracer(__name__)

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute(AGENT_NAME_KEY, name)
                span.set_attribute("openinference.span.kind", "CHAIN")
                if framework:
                    span.set_attribute("agent.framework", framework)
                if extra:
                    for k, v in extra.items():
                        span.set_attribute(k, v)
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def tool_span(fn=None, *, description: str | None = None, extra: dict | None = None) -> Callable:
    """
    Wrap a tool call in a named span with tool metadata, including inputs and outputs.

    Can be used with or without parentheses:
        @tool_span
        @tool
        def my_tool(): ...

    Or with parameters:
        @tool_span(description="My tool description")
        @tool
        def my_tool(): ...

    Args:
        fn: Function being decorated
        description: Optional tool description
        extra: Optional additional span attributes
    """
    tracer = trace.get_tracer(__name__)

    def decorator(func):
        # Extract function name and docstring for description
        span_name = func.name if isinstance(func, Tool) else func.__name__
        func_description = description or (inspect.getdoc(func) or "").split("\n")[0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("openinference.span.kind", "TOOL")
                span.set_attribute("tool.name", span_name)

                if func_description:
                    span.set_attribute("tool.description", func_description)

                if extra:
                    for k, v in extra.items():
                        span.set_attribute(k, v)

                # Capture input
                input_data = {"args": [str(arg) for arg in args], "kwargs": {k: str(v) for k, v in kwargs.items()}}
                span.set_attribute("input.value", json.dumps(input_data, default=str))

                # Execute function and capture output
                result = func(*args, **kwargs)

                # Capture output
                span.set_attribute("output.value", str(result))
                span.set_attribute("output.mime_type", "text/plain")

                return result

        return wrapper

    # Support both @tool_span and @tool_span()
    if fn is None:
        return decorator
    else:
        return decorator(fn)
