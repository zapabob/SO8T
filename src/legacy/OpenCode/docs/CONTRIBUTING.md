# Contributing to SO8T Safe Agent

Thank you for your interest in contributing to SO8T Safe Agent! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** (see below)
4. **Create a feature branch** for your changes
5. **Make your changes** following our guidelines
6. **Test your changes** thoroughly
7. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8+
- CUDA 12.0+ (for GPU support)
- Git
- Docker (optional, for containerized development)

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/so8t-safe-agent.git
   cd so8t-safe-agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

5. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

6. **Run tests to verify setup**
   ```bash
   pytest
   ```

### Docker Development

For containerized development:

```bash
# Build development image
docker build -t so8t-dev --target development .

# Run development container
docker run -it --gpus all -v $(pwd):/app so8t-dev
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix existing issues
- **Features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Security**: Improve security measures

### Before You Start

1. **Check existing issues** to see if your idea is already being discussed
2. **Create an issue** for significant changes to discuss the approach
3. **Read the documentation** to understand the codebase
4. **Review the code style** guidelines

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   pytest
   
   # Run specific tests
   pytest tests/test_your_module.py
   
   # Run with coverage
   pytest --cov=models --cov=training --cov=inference
   ```

4. **Check code quality**
   ```bash
   # Format code
   black .
   isort .
   
   # Lint code
   flake8 .
   mypy .
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Process

### Before Submitting

- [ ] **Tests pass**: All tests must pass
- [ ] **Code coverage**: Maintain or improve test coverage
- [ ] **Documentation**: Update relevant documentation
- [ ] **Type hints**: Add type hints for new functions
- [ ] **Docstrings**: Add docstrings for new functions/classes
- [ ] **Changelog**: Update CHANGELOG.md if applicable

### PR Template

When creating a pull request, please use the following template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Security improvement

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] No breaking changes (or breaking changes documented)

## Related Issues
Closes #issue_number
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in staging environment
4. **Approval** from at least one maintainer
5. **Merge** after approval

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Detailed steps to reproduce
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, CUDA version, etc.
- **Logs**: Relevant error messages or logs
- **Screenshots**: If applicable

### Feature Requests

When requesting features, please include:

- **Description**: Clear description of the feature
- **Use case**: Why this feature is needed
- **Proposed solution**: How you think it should work
- **Alternatives**: Other solutions you've considered
- **Additional context**: Any other relevant information

## Code Style

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Imports**: Use `isort` for import sorting
- **Type hints**: Required for all public functions
- **Docstrings**: Required for all public functions/classes

### Code Formatting

We use automated tools for code formatting:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with underscore `_`

### Example

```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExampleClass:
    """Example class with proper documentation."""
    
    def __init__(self, name: str, value: int = 0):
        """Initialize the example class.
        
        Args:
            name: The name of the instance
            value: The initial value (default: 0)
        """
        self.name = name
        self.value = value
    
    def process_data(self, data: List[str]) -> Dict[str, int]:
        """Process the input data.
        
        Args:
            data: List of strings to process
            
        Returns:
            Dictionary mapping strings to their lengths
            
        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        return {item: len(item) for item in data}
```

## Testing

### Test Structure

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Test performance characteristics

### Test Naming

- **Test files**: `test_*.py` or `*_test.py`
- **Test functions**: `test_*`
- **Test classes**: `Test*`

### Test Coverage

- **Minimum coverage**: 80%
- **New code**: 90%+ coverage required
- **Critical paths**: 95%+ coverage required

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=models --cov=training --cov=inference

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::TestSO8TModel::test_forward

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto
```

## Documentation

### Documentation Standards

- **Docstrings**: Use Google style docstrings
- **Type hints**: Include type hints in docstrings
- **Examples**: Include usage examples where helpful
- **Cross-references**: Link to related functions/classes

### Documentation Structure

- **README.md**: Project overview and quick start
- **docs/**: Detailed documentation
- **examples/**: Usage examples
- **CHANGELOG.md**: Release notes
- **CONTRIBUTING.md**: This file

### Updating Documentation

- **Code changes**: Update relevant docstrings
- **New features**: Update README and docs
- **API changes**: Update API documentation
- **Breaking changes**: Update CHANGELOG.md

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version** in `__init__.py` and `setup.py`
2. **Update CHANGELOG.md** with release notes
3. **Create release branch** from main
4. **Run full test suite** and fix any issues
5. **Update documentation** if needed
6. **Create pull request** for release
7. **Review and merge** release PR
8. **Tag release** on GitHub
9. **Publish to PyPI** (if applicable)

## Getting Help

### Resources

- **Documentation**: Check the docs/ directory
- **Issues**: Search existing issues on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Code review**: Ask for help in PR comments

### Contact

- **Maintainers**: @maintainer1, @maintainer2
- **Email**: maintainers@so8t-safe-agent.com
- **Discord**: [SO8T Safe Agent Discord](https://discord.gg/so8t-safe-agent)

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Mentioned in release notes
- **Documentation**: Credited in relevant sections
- **GitHub**: Listed as contributors

## License

By contributing to SO8T Safe Agent, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to SO8T Safe Agent! 🚀
