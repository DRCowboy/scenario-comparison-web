from setuptools import setup, find_packages

setup(
    name="scenario-comparison-web",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.2",
        "Flask==2.3.3",
        "pandas==2.1.1",
        "openpyxl==3.1.2",
        "gunicorn==21.2.0",
        "Werkzeug==2.3.7",
    ],
    python_requires=">=3.11,<3.12",
) 