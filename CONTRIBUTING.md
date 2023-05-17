# Contributing to KUNAI-static-analyzer

We welcome contributions to KUNAI-static-analyzer! Here are a few guidelines to follow when contributing to the project.

## Branches

The `main` branch is protected, so direct pushes to this branch are not allowed. Instead, please create a branch off of `main` for your changes. If you're adding a new feature, create a branch named `feature/your-feature-name`. If you're fixing a bug, create a branch named `bugfix/your-bug-name`. When you're ready to merge your changes, open a pull request to `main`.
Currently we are working in `refactoring` branch, so you can find the most updated code in there, but that will change once all the features for next release are implemented and are stable enough.


## Coding Style

Please try to adhere to the coding style used in the project to maintain consistency. Here are some guidelines:

- Use `snake_case` for variable and function names.
- Use `PascalCase` for class and struct names.
- Use `UPPERCASE_WITH_UNDERSCORES` for macros.
- Use spaces instead of tabs, with a tab width of 4 spaces.
- Use C++11 style of initializing variables, i.e., `int x{5}` instead of `int x = 5`.
- Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) instead of raw pointers whenever possible.
- Use `nullptr` instead of `NULL`.
- Use include guards in the header files, these include guards must follow the next convention: `KUNAI_<Folder1>_<Folder2>_<FolderN>_<FileName>_HPP`.

## Testing

Before submitting a pull request, please ensure that your changes pass the existing tests and add any additional tests as necessary.
Currently tests are implemented in the folder: `kunai-lib/unit-tests/`, and we are using CMake for creating the tests and running them using **ctest**. If you want to include some DEX file for testing, include its source code into `kunai-lib/tests/`

## Code of Conduct

By participating in this project, you agree to abide by the Code of Conduct.

If you have any questions or issues, please open an issue in the project repository or contact one of the project maintainers.
