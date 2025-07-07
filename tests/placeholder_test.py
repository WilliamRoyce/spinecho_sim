from __future__ import annotations


def test_import() -> None:
    try:
        import spinecho_sim  # noqa: PLC0415
    except ImportError:
        spinecho_sim = None

    assert spinecho_sim is not None, "spinecho_sim module should not be None"
