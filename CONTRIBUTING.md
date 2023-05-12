# Contributing to KUNAI-static-analyzer

We welcome contributions to KUNAI-static-analyzer! Here are a few guidelines to follow when contributing to the project.

## Branches

The `main` branch is protected, so direct pushes to this branch are not allowed. Instead, please create a branch off of `main` for your changes. If you're adding a new feature, create a branch named `feature/your-feature-name`. If you're fixing a bug, create a branch named `bugfix/your-bug-name`. When you're ready to merge your changes, open a pull request to `main`.

## Coding Style

Please try to adhere to the coding style used in the project to maintain consistency. Here are some guidelines:

- Use `snake_case` for variable and function names.
- Use `PascalCase` for class and struct names.
- Use `UPPERCASE_WITH_UNDERSCORES` for macros.
- Use spaces instead of tabs, with a tab width of 4 spaces.
- Use C++11 style of initializing variables, i.e., `int x{5}` instead of `int x = 5`.
- Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) instead of raw pointers whenever possible.
- Use `nullptr` instead of `NULL`.
- Use `#pragma once` instead of include guards.

## Testing

Before submitting a pull request, please ensure that your changes pass the existing tests and add any additional tests as necessary.

## Code of Conduct

By participating in this project, you agree to abide by the Code of Conduct.

If you have any questions or issues, please open an issue in the project repository or contact one of the project maintainers.
