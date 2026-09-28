def test_package_exposes_version():
    import xor_study

    assert xor_study.__version__ == "0.1.0"
