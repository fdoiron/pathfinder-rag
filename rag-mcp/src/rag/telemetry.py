from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    ExponentialHistogramDataPoint,
    HistogramDataPoint,
    MetricsData,
    NumberDataPoint,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from rag.config import Settings


def _format_data_point(dp: NumberDataPoint | HistogramDataPoint | ExponentialHistogramDataPoint) -> str:
    if isinstance(dp, NumberDataPoint):
        return str(dp.value)
    return f'count={dp.count} sum={dp.sum}'


def _compact_metric_formatter(data: MetricsData) -> str:
    lines = [
        f'{m.name}={_format_data_point(dp)}'
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        for dp in m.data.data_points
    ]
    return '\n'.join(lines) + '\n' if lines else ''


_telemetry_configured = False


def configure_telemetry(settings: Settings, service_name: str, with_metrics: bool = False) -> None:
    """Register this process's tracer (and optionally meter) provider once."""
    global _telemetry_configured
    if _telemetry_configured:
        return

    resource = Resource.create({'service.name': service_name})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True))
    )
    trace.set_tracer_provider(tracer_provider)

    if with_metrics:
        metric_readers = []
        if settings.otel_console_export:
            metric_readers.append(
                PeriodicExportingMetricReader(ConsoleMetricExporter(formatter=_compact_metric_formatter))
            )
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=metric_readers))

    _telemetry_configured = True
