import yaml
class Config:

    def __init__(self, path: str):

        with open(path, "r") as f:
            self.data = yaml.safe_load(f)

    @property
    def model_path(self):
        return self.data["model_path"]

    @property
    def model_gesture(self):
        return self.data['model_gesture']

    @property
    def frame_interval(self):
        return self.data["frame_interval"]

    @property
    def fps(self):
        return self.data["fps"]

    @property
    def connections(self):
        return self.data['connections']

    @property
    def sensors(self):
        return self.data['sensors']

    @property
    def dimension_activities(self):
        return {
            item["classes"]: item["activities"]
            for item in self.data["dimension_activities"]
        }