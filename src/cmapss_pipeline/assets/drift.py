import pandas as pd
import dagster as dg
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from ..config import FEATURES
from ..resources.evidentlyai_resource import EvidentlyAIResource


@dg.asset
def drift_report(evidentlyai_resource: EvidentlyAIResource, train_data: pd.DataFrame) -> dict:
    reference = train_data[train_data['RUL'] > 100][FEATURES]
    current = train_data[train_data['RUL'] <= 30][FEATURES]

    ws, ev_proj = evidentlyai_resource.setup()
    schema = DataDefinition(numerical_columns=FEATURES)

    reference_dataset = Dataset.from_pandas(
        data=reference,
        data_definition=schema
    )
    current_dataset = Dataset.from_pandas(
        data=current,
        data_definition=schema
    )
    report = Report(
        [
            DataDriftPreset(),
            DataSummaryPreset()
        ],
        include_tests=True
    )

    snapshot = report.run(reference_dataset, current_dataset)
    ws.add_run(ev_proj.id, snapshot, include_data=False)

    result = snapshot.dict()
    summary = result['metrics'][0]['value']
    return {
        'n_drifted_features': summary['count'],
        'share_drifted': summary['share']
    }
