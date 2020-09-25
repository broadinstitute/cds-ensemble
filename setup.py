from setuptools import setup

setup(
    name="cds-ensemble",
    version="0.1",
    py_modules=[],
    install_requires=["click>=7", "pandas>=1"],
    entry_points="""
        [console_scripts]
        cds-ensemble=cds-ensemble:cli
    """,
)
