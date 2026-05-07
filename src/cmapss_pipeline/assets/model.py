from datetime import datetime

import dagster as dg
import matplotlib.pyplot as plt
import mlflow.sklearn as mlflow_sklearn
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from ..resources.mlflow_resource import MLflowResource


class TuneModelConfig(dg.Config):
    n_iter: int = 10
    cv: int = 3
    random_state: int = 42

class RegisterModelConfig(dg.Config):
    drift_threshold: float = 0.3
    prod_model_name: str = 'cmapss-rul-predictor'

@dg.asset
def tuned_GBR_model(config: TuneModelConfig, mlflow_resource: MLflowResource, split_data: dict) -> dict:
    X_train = split_data['X_train']
    y_train = split_data['y_train']

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=config.random_state))
    ])
    param_distribs = {
        'model__n_estimators': randint(50, 300),
        'model__max_depth': randint(2, 8),
        'model__learning_rate': uniform(0.01, 0.29),
        'model__min_samples_leaf': randint(1, 10),
        'model__subsample': uniform(0.6, 0.4),
    }

    mlflow = mlflow_resource.setup()
    today = datetime.now().strftime('%Y-%m-%d')
    with mlflow.start_run(run_name=f'hyperparameter-tuning/GBR-randomized/{today}'):
        run_id = mlflow.active_run().info.run_id

        rnd_search = RandomizedSearchCV(
            pipeline, param_distribs, n_iter=config.n_iter, scoring='neg_root_mean_squared_error',
            cv=config.cv, random_state=config.random_state
        )
        rnd_search.fit(X_train, y_train)

        mlflow.log_params(rnd_search.best_params_)
        mlflow.log_metrics({
            'best_mean_RMSE': -rnd_search.best_score_,
            'best_std_RMSE': rnd_search.cv_results_['std_test_score'][rnd_search.best_index_]
        })

        mlflow.set_tags({
            'phase': 'hyperparameter-tuning',
            **config.model_dump(),
            'dataset_version': 'v1'
        })

        best_model = rnd_search.best_estimator_
        signature = infer_signature(X_train[:1], rnd_search.best_estimator_.predict(X_train[:1]))
        mlflow_sklearn.log_model(best_model, 'model', signature=signature)

    return {
        'model': best_model,
        'run_id': run_id,
        'date': today
    }

@dg.asset
def evaluation_model(mlflow_resource: MLflowResource, tuned_GBR_model: dict, split_data: dict) -> dict:
    run_id = tuned_GBR_model['run_id']
    date   = tuned_GBR_model['date']

    X_test = split_data['X_test']
    y_test = split_data['y_test']
    model: Pipeline = tuned_GBR_model['model']

    mlflow = mlflow_resource.setup()
    with mlflow.start_run(run_id=run_id):
        y_preds = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_preds)
        mae  = mean_absolute_error(y_test, y_preds)
        r2   = r2_score(y_test, y_preds)

        metrics = {
            'test_RMSE': rmse,
            'test_MAE': mae,
            'test_R2_score': r2,
        }
        mlflow.log_metrics(metrics)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 10))
        ax_1 = axes[0]
        ax_2 = axes[1]
        
        ax_1.scatter(y_test, y_preds, c='crimson', alpha=0.6)
        ax_1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'b-', lw=2)
        ax_1.set_xlabel('True Values')
        ax_1.set_ylabel('Predictions')
        
        residuals = y_preds - y_test
        ax_2.scatter(y_preds, residuals, c='crimson', alpha=0.6)
        ax_2.set_xlabel('Predicted RUL')
        ax_2.set_ylabel('Residual')

        mlflow.log_figure(fig, f'error_analysis_{date}.png')
        plt.close(fig)

    return metrics

@dg.asset
def register_model(config: RegisterModelConfig, mlflow_resource: MLflowResource, tuned_GBR_model: dict, evaluation_model: dict, drift_report: dict) -> dict:
    mlflow = mlflow_resource.setup()
    mlflow_client = MlflowClient(mlflow.get_tracking_uri())

    try:
        champion = mlflow_client.get_model_version_by_alias(config.prod_model_name, alias='champion')
        champion_rmse = mlflow_client.get_run(champion.run_id).data.metrics.get('test_RMSE')
    except:
        champion = None
        champion_rmse = float('inf')

    drift_detected = drift_report['share_drifted'] > config.drift_threshold
    new_model_better = evaluation_model['test_RMSE'] < champion_rmse
    should_register = drift_detected or new_model_better
    if should_register:
        run_id = tuned_GBR_model['run_id']
        registered = mlflow.register_model(f'runs:/{run_id}/model', config.prod_model_name)
        mlflow_client.set_registered_model_alias(config.prod_model_name, 'champion', registered.version)
        return {
            'registered': True,
            'current_model_version': registered.version,
            'reason': 'better_model' if new_model_better else 'drift_detected'
        }

    return {
        'registered': False,
        'current_model_version': champion.version,
    }