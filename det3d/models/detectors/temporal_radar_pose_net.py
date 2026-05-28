from .. import builder
from ..registry import DETECTORS
from .pose_net import PoseNet


@DETECTORS.register_module
class TemporalRadarPoseNet(PoseNet):
    """Radar pose detector with a causal temporal feature neck.

    The expected radar tensor is (B, T, D, Z, Y, X). For convenience, a normal
    per-frame tensor (B, D, Z, Y, X) is treated as a one-frame window.
    """

    def __init__(
        self,
        reader,
        backbone,
        temporal_neck,
        neck,
        pose_head,
        sensor_type="rdr",
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__(
            reader=reader,
            backbone=backbone,
            sensor_type=sensor_type,
            neck=neck,
            pose_head=pose_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=None,
        )
        self.temporal_neck = builder.build_neck(temporal_neck)
        self.init_weights(pretrained=pretrained)

    def extract_feat(self, data):
        rdr_tensor = data["rdr_tensor"]
        if rdr_tensor.dim() == 5:
            rdr_tensor = rdr_tensor.unsqueeze(1)
        if rdr_tensor.dim() != 6:
            raise ValueError(
                "TemporalRadarPoseNet expects rdr_tensor with shape "
                f"(B,T,D,Z,Y,X) or (B,D,Z,Y,X), got {tuple(rdr_tensor.shape)}"
            )

        batch, time = rdr_tensor.shape[:2]
        frame_shape = rdr_tensor.shape[2:]
        x = rdr_tensor.reshape(batch * time, *frame_shape)
        x = self.reader(x)
        x = self.backbone(x)
        if x.dim() != 5:
            raise ValueError(
                "TemporalRadarPoseNet expects the backbone to return "
                f"(B*T,C,Z,Y,X), got {tuple(x.shape)}"
            )

        channels, z, y, x_size = x.shape[1:]
        x = x.reshape(batch, time, channels, z, y, x_size)
        x = self.temporal_neck(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        example_sensor = {}
        example_sensor.update(example[self.sensor_type])
        example_sensor.update({"meta": example["meta"]})
        x = self.extract_feat(example_sensor)
        preds, _ = self.pose_head(x)
        if return_loss:
            return self.pose_head.loss(example_sensor, preds, self.test_cfg)
        return self.pose_head.predict(example_sensor, preds, self.test_cfg)
