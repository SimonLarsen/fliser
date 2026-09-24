from packaging.version import Version


def test_import():
    import fliser

    Version(fliser.__version__)
