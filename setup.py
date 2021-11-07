"""Install Reinforcement Learning via Supervised Learning."""

import setuptools

import src.rvs  # pytype: disable=import-error

TESTS_REQUIRE = [
    "black",
    "coverage",
    "codecov",
    "codespell",
    "darglint",
    "flake8",
    "flake8-blind-except",
    "flake8-builtins",
    "flake8-commas",
    "flake8-debugger",
    "flake8-docstrings",
    "flake8-isort",
    "isort",
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "pytype",
]


def get_readme():
    """Fetch README from file."""
    with open("README.md", "r") as f:
        return f.read()


setuptools.setup(
    name="rvs",
    version=src.rvs.__version__,
    description="Offline RL via Supervised Learning",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    # gcsl's room_world/room_env.py line 18 uses an import statement that is deprecated
    # and won't work in Python 3.9
    # from __future__ import annotations needs Python >=3.7 for postponed evaluation of
    # annotations
    python_requires=">=3.7.0,<3.9",  # if you change this, also update project.toml
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "configargparse",
        "d4rl @ git+https://github.com/rail-berkeley/d4rl.git@master#egg=d4rl",
        "gcsl @ git+https://github.com/scottemmons/gcsl.git@master#egg=gcsl",
        "gym",
        "matplotlib",
        "numpy",
        "pandas",
        "pytorch-lightning",
        "seaborn",
        "stable-baselines3<=1.2.0",  # we use ActorCriticPolicy's _get_latent function
        "torch<=1.7.1",  # gcsl's rlutil breaks with torch==1.8.1
        "tqdm",
        "wandb",
    ],
    tests_require=TESTS_REQUIRE,
    extras_require={
        "test": TESTS_REQUIRE,
    },
    url="https://github.com/scottemmons/rvs",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
