import dagster as dg


drift_job = dg.define_asset_job(
    name='drift_job',
    selection=['drift_report']
)

daily_drift = dg.ScheduleDefinition(
    job=drift_job,
    cron_schedule='0 6 * * *'
)