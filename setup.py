from setuptools import setup, find_packages

setup(
    name="cds-ensemble",
    version="0.2",
    packages=find_packages(include=["cds_ensemble"]),
    install_requires=[
        "click>=7,<8",
        "numpy==1.21.6",
        "pandas==1.3.5",
        "pyarrow==9.0.0",
        "scipy==1.7.3",
        "scikit-learn==1.0.2",
    ],
    entry_points="""
        [console_scripts]
        cds-ensemble=cds_ensemble.__main__:main
    """,
)
