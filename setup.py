from setuptools import setup, find_packages

setup(
    name="cds-ensemble",
    version="0.1",
    packages=find_packages(include=["cds_ensemble"]),
    install_requires=[
        "click>=7,<8",
        "pandas>=1,<2",
        "pyarrow>=3,<4",
        "pyyaml>=5,<6",
        "scikit-learn>=0,<1",
        "typing-extensions>=3,<4",
    ],
    entry_points="""
        [console_scripts]
        cds-ensemble=cds_ensemble.__main__:main
    """,
)
