from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from pathfinder_agent.config import Settings

_telemetry_configured = False


def configure_telemetry(settings: Settings, service_name: str) -> None:
    """Register this process's tracer provider once."""
    global _telemetry_configured
    if _telemetry_configured:
        return

    resource = Resource.create({'service.name': service_name})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True))
    )
    trace.set_tracer_provider(tracer_provider)

    _telemetry_configured = True
