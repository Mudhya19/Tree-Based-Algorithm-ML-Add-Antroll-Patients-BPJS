from setuptools import setup, find_packages

setup(
    name="bpjs-antrol-analysis",
    version="0.1.0",
    description="BPJS Antrol Patients Analysis using Tree-Based Algorithms",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
        "sqlalchemy>=1.4.0",
        "pymysql>=1.0.0",
        "python-dotenv>=0.19.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)