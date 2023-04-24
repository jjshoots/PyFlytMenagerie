"""Setup PyFlytMenagerie."""
import os

from setuptools import setup


def package_files(directory):
    """package_files.

    Args:
        directory: directory of non-python files
    """
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


setup(
    name="PyFlytMenagerie",
    package_data={"PyFlytMenagerie": package_files("PyFlytMenagerie/*/")},
)
