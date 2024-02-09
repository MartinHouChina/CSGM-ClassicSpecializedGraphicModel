import torch.nn
from torch import nn
from EfficientNetTemplate import *
from time import time_ns

def get_tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


class MyEfficientNetB0(torch.nn.Module):
    def __init__(self, num_classes=4, init_weight=True, special_norm_layer=None):
        super(MyEfficientNetB0, self).__init__()
        width_coefficient = 1.0
        depth_coefficient = 1.0
        dropout_rate = 0.2,
        num_classes = num_classes

        if special_norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        else:
            norm_layer = special_norm_layer

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        self.Conv3x3 = ConvBNActivation(in_planes=3,
                                        out_planes=adjust_channels(32),
                                        kernel_size=3,
                                        stride=2,
                                        norm_layer=norm_layer)

        self.MBConv1_2A = InvertedResidual(
            bneck_conf(3, 32, 16, 1, 1, True, 0.0, "2A"),
            norm_layer
        )

        self.MBConv6_3A = InvertedResidual(
            bneck_conf(3, 16, 24, 6, 2, True, 0.0125, "3A"),
            norm_layer
        )
        self.MBConv6_3B = InvertedResidual(
            bneck_conf(3, 24, 24, 6, 1, True, 0.025, "3B"),
            norm_layer
        )

        self.MBConv6_4A = InvertedResidual(
            bneck_conf(5, 24, 40, 6, 2, True, 0.0375, "4A"),
            norm_layer
        )
        self.MBConv6_4B = InvertedResidual(
            bneck_conf(5, 40, 40, 6, 1, True, 0.05, "4B"),
            norm_layer
        )

        self.MBConv6_5A = InvertedResidual(
            bneck_conf(3, 40, 80, 6, 2, True, 0.0625, "5A"),
            norm_layer
        )
        self.MBConv6_5B = InvertedResidual(
            bneck_conf(3, 80, 80, 6, 1, True, 0.075, "5B"),
            norm_layer
        )
        self.MBConv6_5C = InvertedResidual(
            bneck_conf(3, 80, 80, 6, 1, True, 0.0875, "5C"),
            norm_layer
        )

        self.MBConv6_6A = InvertedResidual(
            bneck_conf(5, 80, 112, 6, 1, True, 0.1, "6A"),
            norm_layer
        )
        self.MBConv6_6B = InvertedResidual(
            bneck_conf(5, 112, 112, 6, 1, True, 0.1125, "6B"),
            norm_layer
        )
        self.MBConv6_6C = InvertedResidual(
            bneck_conf(5, 112, 112, 6, 1, True, 0.125, "6C"),
            norm_layer
        )

        self.MBConv6_7A = InvertedResidual(
            bneck_conf(5, 112, 192, 6, 2, True, 0.1375, "7A"),
            norm_layer
        )
        self.MBConv6_7B = InvertedResidual(
            bneck_conf(5, 192, 192, 6, 1, True, 0.15, "7B"),
            norm_layer
        )
        self.MBConv6_7C = InvertedResidual(
            bneck_conf(5, 192, 192, 6, 1, True, 0.1625, "7C"),
            norm_layer
        )
        self.MBConv6_7D = InvertedResidual(
            bneck_conf(5, 192, 192, 6, 1, True, 0.175, "7D"),
            norm_layer
        )
        self.MBConv6_8A = InvertedResidual(
            bneck_conf(3, 192, 320, 6, 1, True, 0.1875, "8A"),
            norm_layer
        )

        last_conv_input_c = 320
        last_conv_output_c = adjust_channels(1280)

        self.Conv1x1 = ConvBNActivation(in_planes=last_conv_input_c,
                                        out_planes=last_conv_output_c,
                                        kernel_size=1,
                                        norm_layer=norm_layer)

        self.Pooling = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Linear(last_conv_output_c, num_classes)

        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        time0 = time_ns()
        mem0 = get_tensor_bytes(x)

        x = self.Conv3x3(x)

        time1 = time_ns()
        mem1 = get_tensor_bytes(x)

        x = self.MBConv1_2A(x)

        time2 = time_ns()
        mem2 = get_tensor_bytes(x)

        x = self.MBConv6_3A(x)
        x = self.MBConv6_3B(x)

        time3 = time_ns()
        mem3 = get_tensor_bytes(x)

        x = self.MBConv6_4A(x)
        x = self.MBConv6_4B(x)

        time4 = time_ns()
        mem4 = get_tensor_bytes(x)

        x = self.MBConv6_5A(x)
        x = self.MBConv6_5B(x)
        x = self.MBConv6_5C(x)

        time5 = time_ns()
        mem5 = get_tensor_bytes(x)

        x = self.MBConv6_6A(x)
        x = self.MBConv6_6B(x)
        x = self.MBConv6_6C(x)

        time6 = time_ns()
        mem6 = get_tensor_bytes(x)

        x = self.MBConv6_7A(x)
        x = self.MBConv6_7B(x)
        x = self.MBConv6_7C(x)
        x = self.MBConv6_7D(x)

        time7 = time_ns()
        mem7 = get_tensor_bytes(x)

        x = self.MBConv6_8A(x)

        time8 = time_ns()
        mem8 = get_tensor_bytes(x)

        x = self.Conv1x1(x)
        x = self.Pooling(x)
        x = torch.flatten(x, 1)
        x = self.FC(x)

        time9 = time_ns()
        mem9 = get_tensor_bytes(x)

        dfn = (
            "0us",
            str((time1 - time0) * (10 ** -6)) + "ms",
            str((time2 - time0) * (10 ** -6)) + "ms",
            str((time3 - time0) * (10 ** -6)) + "ms",
            str((time4 - time0) * (10 ** -6)) + "ms",
            str((time5 - time0) * (10 ** -6)) + "ms",
            str((time6 - time0) * (10 ** -6)) + "ms",
            str((time7 - time0) * (10 ** -6)) + "ms",
            str((time8 - time0) * (10 ** -6)) + "ms",
            str((time9 - time0) * (10 ** -6)) + "ms"
        )

        mem_seq = (
            mem0,
            mem1,
            mem2,
            mem3,
            mem4,
            mem5,
            mem6,
            mem7,
            mem8,
            mem9,
        )

        return x, dfn, mem_seq
