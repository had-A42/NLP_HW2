"""Normalize Kaggle SDK notebook references without changing their identity."""

from urllib.parse import urlparse


def canonical_kernel_ref(value):
    path = urlparse(value).path if "://" in value else value
    parts = path.strip("/").split("/")
    if parts and parts[0] == "code":
        parts = parts[1:]
    if len(parts) not in {2, 3} or not all(parts):
        raise ValueError("Expected an owner/kernel[/version] Kaggle reference.")
    if len(parts) == 3 and not parts[2].isdigit():
        raise ValueError("Notebook version must be numeric.")
    return "/".join(parts)


def versioned_kernel_ref(value, version):
    return "/".join(canonical_kernel_ref(value).split("/")[:2] + [str(version)])
