import dagster as dg

DATASET = 'FD001'


retrain_job = dg.define_asset_job(
    name='retrain_job',
    selection=[
        'train_data', 'test_data', 'split_data', 'tuned_GBR_model', 'evaluation_model'
    ],
    config={
        'ops': {
            'train_data': {
                'config': {'dataset': DATASET}
            },
            'test_data': {
                'config': {'dataset': DATASET}
            },
            'tuned_GBR_model': {
                'config': {'n_iter': 100, 'cv': 10, 'random_state': 42}
            }
        }
    }
)

weekly_retrain = dg.ScheduleDefinition(
    job=retrain_job,
    cron_schedule='0 6 * * 1'
)