# ==============================================================================
# Copyright 2024 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
# https://github.com/NVIDIA/BigVGAN/tree/v2.4

import math
from typing import Optional, Sequence

import torch
import torchaudio
from torch import nn


__all__ = ["BigVGAN"]


class BigVGAN(nn.Module):
    """BigVGAN (see https://arxiv.org/abs/2206.04658).

    Default configuration: nvidia/bigvgan_v2_24khz_100band_256x.

    Arguments
    ---------
    n_fft:
        Number of FFT points for computing the spectrogram.
    n_mels:
        Number of Mel filterbanks.
    sample_rate:
        The sampling rate of the audio signal.
    hop_size:
        The number of audio samples between adjacent frames in the spectrogram.
    win_size:
        The size of the window used for FFT.
    fmin:
        Minimum frequency for the Mel filterbank.
    fmax:
        Maximum frequency for the Mel filterbank. If `None`, it is set to half the sample rate.
    center:
        Whether the input signal is padded so that the FFT window is centered.
    upsample_rates:
        A tuple defining the upsampling factors for each layer.
    upsample_kernel_sizes:
        A tuple defining the kernel sizes for the upsampling layers.
    upsample_initial_channel:
        Number of channels for the first upsampling layer.
    resblock_kernel_sizes:
        A tuple defining the kernel sizes for the residual blocks.
    resblock_dilation_sizes:
        A tuple of tuples defining the dilation sizes for each layer in the residual blocks.

    """

    def __init__(
        self,
        n_fft: int = 1024,
        n_mels: int = 100,
        sample_rate: int = 24000,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: int = 0,
        fmax: Optional[int] = None,
        center: bool = False,
        upsample_rates: Sequence[int] = (4, 4, 2, 2, 2, 2),
        upsample_kernel_sizes: Sequence[int] = (8, 8, 4, 4, 4, 4),
        upsample_initial_channel: int = 1536,
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        resblock_dilation_sizes: Sequence[int] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes

        # Modules
        self.prenet = nn.Conv1d(n_mels, upsample_initial_channel, 7, 1, padding=3)
        self.upsamples = nn.ModuleList()
        self.ampblocks = nn.ModuleList()
        out_channels = None
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_channels = upsample_initial_channel // (2**i)
            out_channels = upsample_initial_channel // (2 ** (i + 1))
            self.upsamples.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )
            self.ampblocks.append(
                nn.ModuleList(
                    AMPBlock1(out_channels, k, d)
                    for (k, d) in zip(resblock_kernel_sizes, resblock_dilation_sizes)
                )
            )
        self.postnet = nn.Sequential(
            UpSample1d(ratio=2, kernel_size=12),
            SnakeBeta(out_channels),
            DownSample1d(ratio=2, kernel_size=12),
            nn.Conv1d(out_channels, 1, 7, 1, padding=3, bias=False),
        )

        # Buffers
        mel_basis = torchaudio.functional.melscale_fbanks(
            int(1 + n_fft // 2),
            fmin,
            sample_rate / 2.0,
            n_mels,
            sample_rate,
            mel_scale="slaney",
            norm="slaney",
        ).T
        hann_window = torch.hann_window(win_size)
        self.register_buffer("mel_basis", mel_basis, persistent=False)
        self.register_buffer("hann_window", hann_window, persistent=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Arguments
        ---------
        feats:
            The input Mel spectrogram features, shape: [B, C, N].

        Returns
        -------
            The output signal, shape: [B, 1, T].

        """
        x = feats
        x = self.prenet(x)

        for i, upsample in enumerate(self.upsamples):
            x = upsample(x)
            ampblocks = self.ampblocks[i]
            y = ampblocks[0](x)
            for resblock in ampblocks[1:]:
                y += resblock(x)
            x = y / len(ampblocks)

        x = self.postnet(x)
        x = x.clamp(min=-1.0, max=1.0)

        return x

    def extract_features(self, sig: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Extract Mel spectrogram features.

        Arguments
        ---------
        sig:
            TThe output signal, shape: [B, T].

        Returns
        -------
            The input Mel spectrogram features, shape: [B, C, N].

        """
        x = sig
        x = torchaudio.functional.resample(x, sample_rate, self.sample_rate)
        return mel_spectrogram(
            x,
            self.mel_basis,
            self.hann_window,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            center=self.center,
        )


class AMPBlock1(torch.nn.Module):
    """Anti-aliased multi-periodicity module (type 1).

    Arguments
    ---------
    channels:
        The number of input and output channels for the convolutional layer.
    kernel_size:
        The size of the convolutional kernel.
    dilation:
        A sequence specifying the dilation rates for the convolution.

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Sequence[int] = (1, 3, 5),
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Modules
        self.resblocks = nn.ModuleList(
            nn.Sequential(
                UpSample1d(ratio=2, kernel_size=12),
                SnakeBeta(channels),
                DownSample1d(ratio=2, kernel_size=12),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=int((kernel_size * d - d) / 2),
                ),
                UpSample1d(ratio=2, kernel_size=12),
                SnakeBeta(channels),
                DownSample1d(ratio=2, kernel_size=12),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=int((kernel_size - 1) / 2),
                ),
            )
            for d in dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resblock in self.resblocks:
            xt = resblock(x)
            x = xt + x
        return x


class SnakeBeta(nn.Module):
    """Snake activation.

    Arguments
    ---------
    channels:
        The number of input channels.

    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Parameters
        self.log_alpha = nn.Parameter(torch.zeros(channels))
        self.log_beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_alpha = self.log_alpha.unsqueeze(dim=-1)  # [C, T]
        log_beta = self.log_beta.unsqueeze(dim=-1)  # [C, T]
        return snake_beta(x, log_alpha, log_beta)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(channels={self.channels})"


class UpSample1d(nn.Module):
    """Upsampling layer.

    Arguments
    ---------
    ratio:
        The factor by which the input is upsampled.
    kernel_size:
        The kernel size. If None, a default kernel size is chosen based on the ratio.

    """

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )

        # Buffers
        upsample_filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("upsample_filter", upsample_filter, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upsample1d(
            x,
            self.upsample_filter,
            self.ratio,
            self.stride,
            self.pad,
            self.pad_left,
            self.pad_right,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ratio={self.ratio}, kernel_size={self.kernel_size})"


class DownSample1d(nn.Module):
    """Downsampling layer.

    Arguments
    ---------
    ratio:
        The factor by which the input is downsampled.
    kernel_size:
        The kernel size. If None, a default kernel size is chosen based on the ratio.

    """

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        downsample_filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=kernel_size,
        )

        # Buffers
        self.register_buffer("downsample_filter", downsample_filter, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return downsample1d(
            x, self.downsample_filter, self.ratio, self.pad_left, self.pad_right
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ratio={self.ratio}, kernel_size={self.kernel_size})"


@torch.jit.script
def mel_spectrogram(
    x: torch.Tensor,
    mel_basis: torch.Tensor,
    hann_window: torch.Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    assert x.min() >= -1.0
    assert x.max() <= 1.0

    padding = (n_fft - hop_size) // 2
    x = torch.nn.functional.pad(x, (padding, padding), mode="reflect")

    spec = torch.stft(
        x,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = ((torch.view_as_real(spec) ** 2).sum(dim=-1) + 1e-9).sqrt()

    mel_spec = mel_basis @ spec
    mel_spec = mel_spec.clamp(min=1e-5).log()
    return mel_spec


@torch.jit.script
def snake_beta(
    x: torch.Tensor, log_alpha: torch.Tensor, log_beta: torch.Tensor
) -> torch.Tensor:
    alpha = log_alpha.exp()
    beta = log_beta.exp()
    x = x + (1.0 / (beta + 1e-9)) * (x * alpha).sin() ** 2
    return x


@torch.jit.script
def upsample1d(
    x: torch.Tensor,
    filter: torch.Tensor,
    ratio: int,
    stride: int,
    pad: int,
    pad_left: int,
    pad_right: int,
) -> torch.Tensor:
    _, C, _ = x.shape
    x = nn.functional.pad(x, (pad, pad), mode="replicate")
    x = ratio * nn.functional.conv_transpose1d(
        x, filter.expand(C, -1, -1), stride=stride, groups=C
    )
    x = x[..., pad_left:-pad_right]
    return x


@torch.jit.script
def downsample1d(
    x: torch.Tensor, filter: torch.Tensor, ratio: int, pad_left: int, pad_right: int
) -> torch.Tensor:
    _, C, _ = x.shape
    x = nn.functional.pad(x, (pad_left, pad_right), mode="replicate")
    x = nn.functional.conv1d(x, filter.expand(C, -1, -1), stride=ratio, groups=C)
    return x


@torch.jit.script
def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> torch.Tensor:  # return filter [1, 1, kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0 * A  # * A required for jitting
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        return torch.zeros_like(time)

    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    # Normalize filter to have sum = 1, otherwise we will have a
    # small leakage of the constant component in the input signal
    filter_ /= filter_.sum()
    filter = filter_.view(1, 1, kernel_size)

    return filter


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BigVGAN().to(device)
    model.eval().requires_grad_(False)
    print(
        f"Total number of parameters: {sum([x.numel() for x in model.state_dict().values()]) / 1e6} M"
    )
    wav_path = "sample.wav"
    sig, orig_sample_rate = torchaudio.load(wav_path)
    sig = sig.to(device)
    with torch.no_grad():
        feats = model.extract_features(sig, orig_sample_rate)
        rec_sig = model(feats).cpu()
    torchaudio.save("reconstruction.wav", rec_sig[0, :], model.sample_rate)
