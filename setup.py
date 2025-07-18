from setuptools import setup, find_packages

setup(
    name="BLIP",  
    version="0.1.0",  
    description="A tool that returns provenances for LLM-powered data processing tasks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
    install_requires=[
        "pdfminer",
        "tiktoken",
        "sklearn",
        "pandas",
        "openai",
        "numpy"
    ],
)
