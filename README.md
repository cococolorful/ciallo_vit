# Template

<div>
    <a href="https://github.com/cococolorful/ciallo_vit/actions/workflows/ci.yaml"><img src="https://github.com/cococolorful/ciallo_vit/actions/workflows/ci.yaml/badge.svg" alt="ciallo_vit CI"></a>
</div>

This is a template library for Python developers. It provides a set of tools and utilities to help you build high-quality Python libraries. For example, it includes a set of unit tests, ~~a documentation generator~~, and a continuous integration system.

## How to use this template library
To use this template library, you need to clone or download it to your local machine. Then, you should modify the name `Template` to your own library name. After that, you can use the library in your Python project.

Suppose you want to create a library called `MyLibrary`. You can follow these steps:
1. Rename the folder `Template` to `MyLibrary`. This folder contains the source code of the your library.
2. Modify some values in the `pyproject.toml`. For example, you can change the name of the library, the version, the dependencies, and so on.
3. Modify this file `README.md`, especially the build passing badge.

## Q&A
### How to install this library?

```shell
python3 -m pip install build
python3 -m build
python3 -m pip install .
```