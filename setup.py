from setuptools import setup

setup(
    name="cds-ensemble",
    version="0.1",
    py_modules=[],
    install_requires=[
        "click>=7",
        "pandas>=1",
        "pyyaml>=5",
        "scikit-learn>=0",
        "typing-extensions>=3",
    ],
    entry_points="""
        [console_scripts]
        cds-ensemble=cds_ensemble.__main__:main
    """,
)
