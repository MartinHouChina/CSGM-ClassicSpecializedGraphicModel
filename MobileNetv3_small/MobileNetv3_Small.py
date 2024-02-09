from torch import nn
from time import time_ns
from MobileNetv3_Official import *
import torch


def get_tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


class MobileNetSmall(nn.Module):
    def __init__(self, num_classes=4, special_norm_layer=None, init_weight=True):
        super(MobileNetSmall, self).__init__()
        width_multi = 1.0
        bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)
        if special_norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        else:
            norm_layer = special_norm_layer

        firstconv_output_c = bneck_conf(16, 3, 16, 16, True, 'RE', 2).input_c

        self.Conv2d_3x3 = ConvBNActivation(3,
                                           firstconv_output_c,
                                           kernel_size=3,
                                           stride=2,
                                           norm_layer=norm_layer,
                                           activation_layer=nn.Hardswish)

        self.bneck_3x3_A = InvertedResidual(
            bneck_conf(16, 3, 16, 16, True, 'RE', 2),
            norm_layer
        )

        self.bneck_3x3_B = InvertedResidual(
            bneck_conf(16, 3, 72, 24, False, 'RE', 2),
            norm_layer
        )

        self.bneck_3x3_C = InvertedResidual(
            bneck_conf(24, 3, 88, 24, False, 'RE', 1),
            norm_layer
        )

        self.bneck_5x5_A = InvertedResidual(
            bneck_conf(24, 5, 96, 40, True, 'HS', 2),
            norm_layer
        )

        self.bneck_5x5_B = InvertedResidual(
            bneck_conf(40, 5, 240, 40, True, 'HS', 1),
            norm_layer
        )

        self.bneck_5x5_C = InvertedResidual(
            bneck_conf(40, 5, 240, 40, True, 'HS', 1),
            norm_layer
        )

        self.bneck_5x5_D = InvertedResidual(
            bneck_conf(40, 5, 120, 48, True, 'HS', 1),
            norm_layer
        )

        self.bneck_5x5_E = InvertedResidual(
            bneck_conf(48, 5, 144, 48, True, 'HS', 1),
            norm_layer
        )

        self.bneck_5x5_F = InvertedResidual(
            bneck_conf(48, 5, 288, 96, True, 'HS', 2),
            norm_layer
        )

        self.bneck_5x5_G = InvertedResidual(
            bneck_conf(96, 5, 576, 96, True, 'HS', 1),
            norm_layer
        )

        self.bneck_5x5_H = InvertedResidual(
            bneck_conf(96, 5, 576, 96, True, 'HS', 1),
            norm_layer
        )

        lastconv_input_c = bneck_conf(96, 5, 576, 96, True, 'HS', 1).out_c
        lastconv_output_c = 6 * lastconv_input_c

        self.Conv2d_1x1 = ConvBNActivation(lastconv_input_c,
                                           lastconv_output_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=nn.Hardswish)

        self.Pooling = nn.AdaptiveAvgPool2d(1)

        last_channel = adjust_channels(1024)

        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
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

        x = self.Conv2d_3x3(x)

        time1 = time_ns()
        mem1 = get_tensor_bytes(x)

        x = self.bneck_3x3_A(x)

        time2 = time_ns()
        mem2 = get_tensor_bytes(x)

        x = self.bneck_3x3_B(x)

        time3 = time_ns()
        mem3 = get_tensor_bytes(x)

        x = self.bneck_3x3_C(x)

        time4 = time_ns()
        mem4 = get_tensor_bytes(x)

        x = self.bneck_5x5_A(x)

        time5 = time_ns()
        mem5 = get_tensor_bytes(x)

        x = self.bneck_5x5_B(x)

        time6 = time_ns()
        mem6 = get_tensor_bytes(x)

        x = self.bneck_5x5_C(x)

        time7 = time_ns()
        mem7 = get_tensor_bytes(x)

        x = self.bneck_5x5_D(x)

        time8 = time_ns()
        mem8 = get_tensor_bytes(x)

        x = self.bneck_5x5_E(x)

        time9 = time_ns()
        mem9 = get_tensor_bytes(x)

        x = self.bneck_5x5_F(x)

        time10 = time_ns()
        mem10 = get_tensor_bytes(x)

        x = self.bneck_5x5_G(x)

        time11 = time_ns()
        mem11 = get_tensor_bytes(x)

        x = self.bneck_5x5_H(x)

        time12 = time_ns()
        mem12 = get_tensor_bytes(x)

        x = self.Conv2d_1x1(x)

        time13 = time_ns()
        mem13 = get_tensor_bytes(x)

        x = self.Pooling(x)

        time14 = time_ns()
        mem14 = get_tensor_bytes(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        time15 = time_ns()
        mem15 = get_tensor_bytes(x)

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
            str((time9 - time0) * (10 ** -6)) + "ms",
            str((time10 - time0) * (10 ** -6)) + "ms",
            str((time11 - time0) * (10 ** -6)) + "ms",
            str((time12 - time0) * (10 ** -6)) + "ms",
            str((time13 - time0) * (10 ** -6)) + "ms",
            str((time14 - time0) * (10 ** -6)) + "ms",
            str((time15 - time0) * (10 ** -6)) + "ms"
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
            mem10,
            mem11,
            mem12,
            mem13,
            mem14,
            mem15
        )

        return x, dfn, mem_seq
