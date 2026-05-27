from setuptools import setup, find_packages

setup(
    name="pysyrev",
    version="0.1.3",
    description="LLM-assisted PRISMA workflow for systematic literature review",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Benjamin Pillot",
    author_email="benjamin.pillot@ird.fr",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["old_version*", "data*", "docs*"]),
    package_data={
        "pysyrev": [
            "config_examples/*.yaml",
            "templates/*.yaml",
            "core/berteley/*.csv",
        ],
    },
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "pyyaml>=6.0",
        "python-dotenv>=1.0",
        "rapidfuzz>=3.0",
        "requests>=2.28",
        "tqdm>=4.64",
        "nest-asyncio>=1.5",
        "pydantic>=2.0",
        "litellm>=1.0",
        "networkx>=3.0",
        "reportlab>=4.0",
        "bertopic>=0.16",
        "hdbscan>=0.8",
        "umap-learn>=0.5",
        "sentence-transformers>=2.0",
        "gensim>=4.3.0",
        "octis>=1.0",
        "nltk>=3.8",
        "spacy>=3.0",
        "beautifulsoup4>=4.11",
        "contractions>=0.1",
    ],
    extras_require={
        "plotly": [
            "plotly>=5.0",
            "kaleido>=0.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "pysyrev=pysyrev.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Science/Research",
    ],
)
