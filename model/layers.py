import torch
import torch.nn as nn
import torch.nn.functional as F

##################################################################################
# Basic Blocks
##################################################################################

class LinearBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout_rate: float = 0.0,
        norm="none",
        activation="none",
    ):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        # if norm == "sn":
        #     self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        # else:
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        assert (
            dropout_rate == 0.0 or norm == "none"
        ), "Mixing Dropout and Norm is a wrong usage."

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        # initialize normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == "none" or norm == "sn":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}" % norm

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)

        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        dropout_rate: float = 0.0,
        norm="none",
        activation="none",
        pad_type="zeros",
    ):
        super(Conv1dBlock, self).__init__()

        assert (
            dropout_rate == 0.0 or norm == "none"
        ), "Mixing Dropout and Norm is a wrong usage."

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # initialize normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == "in":
            # self.norm = nn.InstanceNorm1d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(norm_dim)
        # elif norm == "adain":
        #     self.norm = AdaptiveInstanceNorm1d(norm_dim)
        # elif norm == "none" or norm == "sn":
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.01)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        # if norm == "sn":
        #     self.conv = SpectralNorm(
        #         nn.Conv1d(
        #             input_dim, output_dim, kernel_size, stride, bias=self.use_bias
        #         )
        #     )
        # else:
        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
            padding=padding,
            padding_mode=pad_type,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            out = self.dropout(out)
        return x


class Conv1dTransposeBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
        norm="none",
        activation="none",
    ):
        super(Conv1dTransposeBlock, self).__init__()
        self.use_bias = True

        # initialize normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(norm_dim)
        # elif norm == 'adain':
        #     self.norm = AdaptiveInstanceNorm1d(norm_dim)
        # elif norm == 'none' or norm == 'sn':
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.01)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        # if norm == 'sn':
        #     self.conv = SpectralNorm(nn.ConvTranspose1d(input_dim, output_dim, kernel_size, stride, padding=padding, output_padding=output_padding, bias=self.use_bias))
        # else:
        self.conv = nn.ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
            padding=padding,
            output_padding=output_padding,
            bias=self.use_bias,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        dropout_rate: float = 0.0,
        norm="none",
        activation="none",
        pad_type="zeros",
    ):
        super(Conv2dBlock, self).__init__()

        assert (
            dropout_rate == 0.0 or norm == "none"
        ), "Mixing Dropout and Norm is a wrong usage."

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # initialize normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm(norm_dim)
        # elif norm == "none" or norm == "sn":
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.01)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        # if norm == "sn":
        #     self.conv = SpectralNorm(
        #         nn.Conv2d(
        #             input_dim, output_dim, kernel_size, stride, bias=self.use_bias
        #         )
        #     )
        # else:
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
            padding=padding,
            padding_mode=pad_type,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            out = self.dropout(out)
        return x


class Conv2dTransposeBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
        norm="none",
        activation="none",
    ):
        super(Conv2dTransposeBlock, self).__init__()
        self.use_bias = True

        # initialize normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm(norm_dim)
        # elif norm == 'none' or norm == 'sn':
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.01)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        # if norm == 'sn':
        #     self.conv = SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, output_padding=output_padding, bias=self.use_bias))
        # else:
        self.conv = nn.ConvTranspose2d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
            padding=padding,
            output_padding=output_padding,
            bias=self.use_bias,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        weight_layer: nn.Module,
        version: int,
        norm: nn.Module = None,
        activation: str = "none",
        shortcut_project: nn.Module = None,
    ):
        super(ResBlock, self).__init__()
        assert version > 0 and version < 3
        self.version = version
        self.weight_layer = weight_layer
        self.shortcut_project = shortcut_project

        self.norm = norm

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.01)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        # if norm == 'sn':
        #     self.conv = SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, output_padding=output_padding, bias=self.use_bias))
        # else:

    def forward(self, x):
        res = self.shortcut_project(x) if self.shortcut_project else x
        if self.version == 1:
            x = self.weight_layer(x)
            if self.norm:
                x = self.norm(x)
            x += res
            if self.activation:
                x = self.activation(x)
        else:
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
            x = self.weight_layer(x)
            x += res
        return x


# def l2normalize(v, eps=1e-12):
#     return v / (v.norm() + eps)


# class SpectralNorm(nn.Module):
#     """
#     Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
#     and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
#     """

#     def __init__(self, module, name="weight", power_iterations=1):
#         super(SpectralNorm, self).__init__()
#         self.module = module
#         self.name = name
#         self.power_iterations = power_iterations
#         if not self._made_params():
#             self._make_params()

#     def _update_u_v(self):
#         u = getattr(self.module, self.name + "_u")
#         v = getattr(self.module, self.name + "_v")
#         w = getattr(self.module, self.name + "_bar")

#         height = w.data.shape[0]
#         for _ in range(self.power_iterations):
#             v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
#             u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

#         # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
#         sigma = u.dot(w.view(height, -1).mv(v))
#         setattr(self.module, self.name, w / sigma.expand_as(w))

#     def _made_params(self):
#         try:
#             u = getattr(self.module, self.name + "_u")
#             v = getattr(self.module, self.name + "_v")
#             w = getattr(self.module, self.name + "_bar")
#             return True
#         except AttributeError:
#             return False

#     def _make_params(self):
#         w = getattr(self.module, self.name)

#         height = w.data.shape[0]
#         width = w.view(height, -1).data.shape[1]

#         u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
#         v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
#         u.data = l2normalize(u.data)
#         v.data = l2normalize(v.data)
#         w_bar = nn.Parameter(w.data)

#         del self.module._parameters[self.name]

#         self.module.register_parameter(self.name + "_u", u)
#         self.module.register_parameter(self.name + "_v", v)
#         self.module.register_parameter(self.name + "_bar", w_bar)

#     def forward(self, *args):
#         self._update_u_v()
#         return self.module.forward(*args)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.is_assigned = False
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor):
        assert self.is_assigned, "Please assign weight and bias before calling AdaIN!"
        batch_size = x.size(0)
        running_mean = self.running_mean.repeat(batch_size)
        running_var = self.running_var.repeat(batch_size)

        # Apply instance norm
        x_reshaped = x.reshape(1, -1, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )
        out = out.reshape(batch_size, -1, *x.size()[2:])
        self.is_assigned = False
        return out

    def assign_params(self, param: torch.Tensor):
        # assert param.size(1) == 2 * self.num_features
        self.is_assigned = True
        self.weight = param[:, : self.num_features].reshape(-1)
        self.bias = param[:, self.num_features :].reshape(-1)

    @staticmethod
    def get_num_adain_params(model: nn.Module):
        num_adain_params = 0
        for m in model.modules():
            if isinstance(m, AdaptiveInstanceNorm):
                num_adain_params += 2 * m.num_features
        return num_adain_params
