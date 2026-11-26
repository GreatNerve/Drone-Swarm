"""One flag picks the compute device: auto, cpu, cuda or mps."""
DEVICES = ("auto", "cpu", "cuda", "mps")


def resolve(name="auto"):
    import torch
    if name not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}")
    mps = getattr(torch.backends, "mps", None)
    has_mps = bool(mps and mps.is_available())
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch build has no CUDA (install a CUDA wheel of torch)")
    if name == "mps" and not has_mps:
        raise ValueError("MPS was requested but is not available on this machine")
    return torch.device(name)


def describe():
    import platform
    import torch
    lines = [f"python {platform.python_version()} on {platform.system()} {platform.machine()}",
             f"torch {torch.__version__} (CUDA build: {torch.version.cuda or 'none'})",
             f"cuda available: {torch.cuda.is_available()}"
             + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""),
             f"mps available: {bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())}",
             f"auto device: {resolve('auto')}"]
    return "\n".join(lines)
