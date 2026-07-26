import yaml
class Config:

    def __init__(self, path: str):

        with open(path, "r") as f:
            self.data = yaml.safe_load(f)

    @property
    def model_path(self):
        return self.data["model_path"]

    @property
    def frame_interval(self):
        return self.data["frame_interval"]

    @property
    def fps(self):
        return self.data["fps"]