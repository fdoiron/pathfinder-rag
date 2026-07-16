import json

from google import genai
from google.oauth2 import service_account

from rag.config import Settings

_SCOPES = ['https://www.googleapis.com/auth/cloud-platform']


def make_genai_client(settings: Settings) -> genai.Client:
    """Builds Vertex AI client, resolving credentials at point of use

    fall back to application default credentials when no service account file configed
    project id comes from settings or the service account file.
    """
    credentials = None
    project = settings.gcp_project

    sa_file = settings.gcp_service_account_file
    if sa_file and sa_file.exists():
        if not project:
            project = json.loads(sa_file.read_text(encoding='utf-8'))['project_id']
        # google-auth's from_service_account_file is untyped upstream despite shipping py.typed
        credentials = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            str(sa_file), scopes=_SCOPES
        )

    if not project:
        raise ValueError('gcp_project must be set via RAG_GCP_PROJECT or a GCP service account file')

    return genai.Client(
        vertexai=True,
        project=project,
        location=settings.gcp_location,
        credentials=credentials,
    )
