from dagster import ConfigurableResource


class EvidentlyAIResource(ConfigurableResource):
    url: str
    api: str
    project_name: str | None = None
    project_id: str | None = None
    org_id: str | None = None

    def setup(self):
        from evidently.ui.workspace import CloudWorkspace
        ws = CloudWorkspace(token=self.api, url=self.url)
        if self.project_name is None and self.project_id is None:
            raise ValueError('One of the attributes: [project_name, project_id] must be specified')
        if self.project_id is not None:
            return ws, ws.get_project(self.project_id)
        return ws, ws.create_project(self.project_name, org_id=self.org_id)