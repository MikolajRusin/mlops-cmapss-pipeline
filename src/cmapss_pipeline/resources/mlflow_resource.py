from dagster import ConfigurableResource


class MLflowResource(ConfigurableResource):
    tracking_uri: str
    experiment_name: str
    
    def setup(self):
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        return mlflow