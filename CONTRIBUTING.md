# Contributing to OpenFabrik

Thank you for your interest in contributing to OpenFabrik! We welcome contributions from the community to make synthetic data generation more accessible and powerful.

## 🎯 Ways to Contribute

### 1. Report Bugs

Found a bug? Please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
- Error messages or logs

### 2. Suggest Features

Have an idea? We'd love to hear it! Open an issue with:
- Clear description of the feature
- Use case and benefits
- Examples of how it would work
- Any relevant research or references

### 3. Improve Documentation

Documentation improvements are always welcome:
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve code comments
- Translate documentation

### 4. Submit Code

Want to contribute code? Great! Please follow these guidelines:

## 🛠️ Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/OpenFabrik.git
cd OpenFabrik

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Install pre-commit hooks
pre-commit install
```

## 📝 Code Guidelines

### Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

### Testing

- Write tests for new features
- Ensure existing tests pass: `pytest tests/`
- Aim for good test coverage
- Test edge cases and error conditions

### Commits

- Use clear, descriptive commit messages
- Follow conventional commits format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `test:` for test additions/changes
  - `chore:` for maintenance tasks

Example:
```
feat: add support for custom augmentation pipelines

- Implement AugmentationPipeline class
- Add configuration file support
- Update documentation with examples
```

## 🔄 Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, documented code
   - Add tests
   - Update documentation

3. **Test your changes**:
   ```bash
   pytest tests/
   python -m flake8 .
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**:
   - Fill out the PR template
   - Link related issues
   - Describe your changes clearly
   - Add screenshots/examples if applicable

6. **Address review feedback**:
   - Respond to comments
   - Make requested changes
   - Push updates to your branch

7. **Merge**:
   - Once approved, your PR will be merged
   - Delete your branch after merge

## 🧪 Testing

We use pytest for testing:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipelines.py

# Run with coverage
pytest --cov=openfabrik tests/
```

## 📚 Documentation

Documentation is built with Sphinx:

```bash
cd docs
make html
# Open docs/_build/html/index.html
```

## 🤔 Questions?

- Open a discussion on GitHub Discussions
- Check existing issues and documentation
- Reach out to maintainers

## 🌟 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Celebrated in our community!

## 📄 License

By contributing to OpenFabrik, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making OpenFabrik better! 🎉
