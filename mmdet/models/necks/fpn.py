import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import auto_fp16
from ..builder import NECKS


@NECKS.register_module()
class FPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            if i<2:
                fpn_conv_cls = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                fpn_conv_local = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv_cls)
                self.fpn_convs.append(fpn_conv_local)
            else:
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(fpn_conv)


            self.lateral_convs.append(l_conv)
            # self.fpn_convs.append(fpn_conv)

        self.fpn_se = nn.ModuleList()
        for i in range(4):
            self.fpn_se.append(nn.Conv2d(2, 2, kernel_size=1))



        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # laterals_1 = laterals

        # build top-down path
        used_backbone_levels = len(laterals)


        laterals_1 = (2*used_backbone_levels - 1)*['']
        laterals_1[2*used_backbone_levels - 2] = laterals[-1]


        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)

            else:
                # prev_shape = laterals[i - 1].shape[2:]
                # laterals[i - 1] += F.interpolate(
                #     laterals[i], size=prev_shape, **self.upsample_cfg)
                # # if i == 2:
                # #     laterals[i - 1] = self.fpn_convs[i-1](laterals[i-1])

                # 注意力机制改进
                prev_shape = laterals[i - 1].shape[2:]
                upsample = F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

                # forward 还能定义池化层吗
                # batch = upsample.shape[0]
                # upsample_gvp = nn.AdaptiveAvgPool2d(1)(upsample)
                # laterals_gvp = nn.AdaptiveAvgPool2d(1)(laterals[i-1])

                upsample_reshape = upsample.reshape(upsample.shape[0], -1)
                laterals_reshape = laterals[i-1].reshape(laterals[i-1].shape[0], -1)

                upsample_mean = upsample_reshape.mean(1, keepdim=True)
                laterals_mean = laterals_reshape.mean(1, keepdim=True)

                input_trans = torch.cat((upsample_mean, laterals_mean), 1)
                weight_local = self.fpn_se[i*2-1](input_trans)
                weight_cls = self.fpn_se[i*2-2](input_trans)

                upsample_local = weight_local[:, 0].unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1).expand_as(upsample)
                lateral_local = weight_local[:, 1].unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1).expand_as(laterals[i-1])
                upsample_cls = weight_cls[:, 0].unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1).expand_as(upsample)
                lateral_cls = weight_cls[:, 0].unsqueeze_(-1).unsqueeze_(-1).unsqueeze_(-1).expand_as(laterals[i-1])

                feature_local = upsample*upsample_local + laterals[i-1]*lateral_local
                feature_cls = upsample*upsample_cls + laterals[i-1]*lateral_cls

                laterals_1[i*2 - 2] = feature_cls
                laterals_1[i*2 - 1] = feature_local


        # build outputs
        # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]

        outs = [
            self.fpn_convs[i](laterals_1[i]) for i in range(used_backbone_levels*2 -1)
        ]

        # outs = []
        # for i in range(used_backbone_levels):
        #     if i != 1:
        #         outs.append(self.fpn_convs[i](laterals[i]))
        #     else:
        #         outs.append(laterals[i])
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        # return tuple(outs)
        return [[outs[0], outs[1]], [outs[2], outs[3]], [outs[4], outs[4]], [outs[5], outs[5]], [outs[6], outs[6]]]
        # return [[laterals_1[0], outs[0], 1], [laterals_1[1], outs[1], 1], [outs[2], outs[2], 0], [outs[3], outs[3], 0], [outs[4], outs[4], 0]]
        # final_outs = []
        # final_outs.extend(laterals[:2])
        # final_outs.extend(outs[2:])
        # final_outs.extend(outs)
        # return tuple(final_outs)
