from setuptools import setup, find_packages

setup(
    name="scenario-comparison-web",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.21.6",
        "Flask==2.0.1",
        "pandas==1.3.5",
        "openpyxl==3.0.10",
        "gunicorn==20.1.0",
        "Werkzeug==2.0.3",
    ],
    python_requires=">=3.9,<3.10",
) 