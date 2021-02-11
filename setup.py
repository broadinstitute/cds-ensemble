from setuptools import setup, find_packages

setup(
    name="cds-ensemble",
    version="0.1",
    packages=find_packages(include=["cds_ensemble"]),
    install_requires=[
        "click~=7",
        "pandas~=1",
        "pyarrow~=3",
        "pyyaml~=5",
        "scikit-learn~=0",
        "typing-extensions~=3",
    ],
    entry_points="""
        [console_scripts]
        cds-ensemble=cds_ensemble.__main__:main
    """,
)
