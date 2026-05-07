from pathlib import Path
from dotenv import load_dotenv

import dagster as dg
from dagster import EnvVar

from .assets.data import train_data, test_data, split_data
from .assets.model import tuned_GBR_model, evaluation_model, register_model
from .assets.drift import drift_report
from .resources.mlflow_resource import MLflowResource
from .resources.evidentlyai_resource import EvidentlyAIResource

# ROOT_DIR = Path(__file__).parents[2]
# load_dotenv(ROOT_DIR / '.env.dev', override=True)


defs = dg.Definitions(
    assets=[
        train_data, test_data, split_data, 
        tuned_GBR_model, evaluation_model, register_model,
        drift_report
    ],
    resources={
        'mlflow_resource': MLflowResource(
            tracking_uri    = EnvVar('MLFLOW_TRACKING_URI'),
            experiment_name = EnvVar('EXPERIMENT_NAME')
        ),
        'evidentlyai_resource': EvidentlyAIResource(
            url          = EnvVar('EVIDENTLY_AI_URL'),
            api          = EnvVar('EVIDENTLY_AI_API_KEY'),
            project_id   = EnvVar('EVIDENTLY_AI_PROJECT_ID'),
            org_id       = EnvVar('EVIDENTLY_AI_ORG_ID')   
        )
    }
)