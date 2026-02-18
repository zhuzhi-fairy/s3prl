from .expert import UpstreamExpert


def streaming_wavlm(*args, **kwargs):
    return UpstreamExpert(*args, **kwargs)
