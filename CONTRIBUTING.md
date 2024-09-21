# Contributing to MyTorch

Thank you for considering contributing to the MyTorch project! This guide will help you get started with setting up your development environment, coding conventions, and the process for submitting changes.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Coding Standards](#coding-standards)
3. [Writing Tests](#writing-tests)
4. [Submitting Changes](#submitting-changes)
5. [License](#license)

## How to Contribute

Contributions can be made in many forms:

- Reporting bugs
- Proposing new features
- Submitting documentation updates
- Writing tests for new or existing functionality
- Fixing bugs or implementing new features

If you're unsure of where to start, check out the "Issues" section on our GitHub repository. Look for issues tagged with "good first issue" if you're new to the project.

## Coding Standards

Please follow these guidelines when contributing:

1. **Naming Conventions:**
   - Use `snake_case` for Python variables and function names.
   - Use `CamelCase` for C++ class names.
   - Use `snake_case` for C++ variable and function names.

2. **Code Formatting:**
   - Use `clang-format` for formatting C++ code. We follow a style similar to the Google style guide.
   - Use `black` for formatting Python code to maintain consistency.

3. **Documentation:**
   - Ensure that your code is well-documented using Python docstrings and C++ comments.
   - If you introduce new functionality, update the project documentation accordingly.

4. **Commits:**
   - Write clear and concise commit messages.
   - Use present tense ("Add feature" not "Added feature").
   - Group related changes into a single commit for better tracking.

## Writing Tests

Testing is critical to maintaining code quality. Please ensure your code is thoroughly tested before submitting.

1. **Python Tests:**
   - Use `unittest` or `pytest` for Python functionality tests.
   - Place Python test files in the `tests/` directory.

2. **C++ Tests:**
   - Use GoogleTest for C++ testing.
   - Place C++ test files in the `csrc/tests/` directory.
   - Regularly run tests to ensure nothing breaks.

## Submitting Changes

Once your changes are ready, follow these steps to submit them:

1. **Fork the Repository:**
   - Click the "Fork" button on GitHub to create a copy of the project in your own GitHub account.

2. **Create a Feature Branch:**
   - Avoid working directly on the `main` branch. Create a new branch for your feature or bug fix with a descriptive name.
  
   ```bash
   git checkout -b feature-add-new-feature
   ```

3. **Commit Your Changes:**

   Make sure your changes adhere to the coding standards mentioned above.
   Write meaningful commit messages that explain the intent of the change.
  
   ```bash
   git add .
   git commit -m "Add new feature to support X"
   ```

4. **Push Your Branch:**
   
   ```bash
   git push origin feature-add-new-feature
   ```

5. **Submit a Pull Request (PR):**
   
   Go to your forked repository on GitHub.
   Click "New Pull Request" and select your feature branch as the source.
   Provide a detailed description of your changes and reference any issues that are resolved by your PR.

6. **Review Process:**
  
   Your pull request will be reviewed by the maintainers.
   If changes are requested, make the necessary updates and push them to your branch. The pull request will update automatically.

## License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.

Thank you for your contributions to MyTorch!

