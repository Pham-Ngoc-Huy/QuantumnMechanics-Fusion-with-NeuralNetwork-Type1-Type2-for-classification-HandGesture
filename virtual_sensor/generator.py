from virtual_sensor.geometry import (
    AngleBetweenCosine,
    ProjectToPalmPlane
)
from virtual_sensor.kinematics import (
    FlexionAngle,
    AbductionAngle,
    WristAngle
)
from config import Config
class GenerateVirtualSensor:
    def __init__(self):
        self.angle = AngleBetweenCosine()
        self.projector = ProjectToPalmPlane()
        self.flexion = FlexionAngle(
            angle_calculator=self.angle
        )
        self.abduction = AbductionAngle(
            projector=self.projector,
            angle_calculator=self.angle
        )
        self.wrist = WristAngle(
            projector=self.projector,
            angle_calculator=self.angle
        )
        self.sensors = Config(path='config/config.yaml').sensors

    def generate(self, landmarks, palm_normal):

        sensor = {}

        for key, cfg in self.sensors.items():
            if not cfg.get('enabled', True):
                continue

            sensor_type = cfg['type']
            parent = cfg.get('parent')
            joint = cfg['joint']
            child = cfg.get('child')

            if sensor_type == 'flexion':
                sensor[key] = self.flexion.calculate(
                    landmarks, (parent, joint, child)
                )
            elif sensor_type == 'abduction':
                ref = cfg['ref_finger']
                sensor[key] = self.abduction.calculate(
                    landmarks,
                    palm_normal,
                    (parent, joint, child),
                    tuple(ref)
                )
            # elif sensor_type == 'wrist_flexion':
            #     sensor[key] = self.wrist.calculate(
            #         landmarks, palm_normal, mode='flexion'
            #     )
            # elif sensor_type == 'wrist_abduction':
            #     sensor[key] = self.wrist.calculate(
            #         landmarks, palm_normal, mode='abduction'
            #     )
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")

        return sensor