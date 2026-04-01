
import torch

EPS = 1e-8


def save_audio(path: str, audio: torch.Tensor, sample_rate: int, fmt: str = "wav") -> None:
    """Save an audio tensor to a file.

    Uses soundfile for wav/flac (avoids the torchcodec requirement in newer
    torchaudio versions). Falls back to torchaudio for mp3 or other formats.

    Args:
        path:        Destination file path.
        audio:       Tensor of shape ``(channels, frames)``, int16 or float32.
        sample_rate: Sample rate in Hz.
        fmt:         File format — ``"wav"``, ``"flac"``, or ``"mp3"``.
    """
    if fmt in ("wav", "flac"):
        import soundfile as sf
        data = audio.numpy().T  # soundfile expects (frames, channels)
        sf.write(str(path), data, sample_rate)
    else:
        import torchaudio
        torchaudio.save(str(path), audio, sample_rate, format=fmt)


def is_silence(audio: torch.Tensor, thresh: float = -60.):
    """checks if entire clip is 'silence' below some dB threshold

    Args:
        audio (Tensor): torch tensor of (multichannel) audio
        thresh (float): threshold in dB below which we declare to be silence
    """

    def get_dbmax(audio: torch.Tensor):
        """finds the loudest value in the entire clip and puts that into dB (full scale)"""
        return 20 * torch.log10(torch.flatten(audio.abs()).max() + EPS).item()

    return get_dbmax(audio) < thresh


def float_to_int16_audio(x: torch.Tensor, maximize: bool = False):
    div = x.abs().max().item()
    if not maximize:
        div = max(div, 1.0)

    return x.div(div).mul(32767).to(torch.int16).cpu()
