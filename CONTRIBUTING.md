# Contributing to PyRamEx

Thank you for your interest in contributing to PyRamEx! This document provides guidelines for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community

## How to Contribute

### Reporting Bugs

1. Check existing issues
2. Create a new issue with:
   - Clear title
   - Detailed description
   - Minimal reproducible example
   - Environment details (Python version, OS, etc.)

### Suggesting Features

1. Check existing issues and discussions
2. Create a feature request with:
   - Clear use case
   - Proposed solution
   - Alternative approaches considered

### Submitting Pull Requests

1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Create a Pull Request

## Development Guidelines

### Code Style

- Use **Black** for formatting (configured in pyproject.toml)
- Use **flake8** for linting
- Use **mypy** for type checking
- Follow PEP 8 guidelines

### Testing

- Write unit tests for new features
- Aim for >= 80% test coverage
- Use pytest framework
- Place tests in `tests/` directory

### Documentation

- Add docstrings to all functions/classes
- Use NumPy docstring format
- Update README.md if user-facing changes
- Add examples for new features

### Commit Messages

- Use clear, descriptive commit messages
- Start with verb (e.g., "Add", "Fix", "Update")
- Include issue number (e.g., "Fix #123")

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/pyramex.git
cd pyramex

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run linting
black pyramex/
flake8 pyramex/
mypy pyramex/
```

## Project Structure

```
pyramex/
├── pyramex/           # Source code
│   ├── core/         # Core data structures
│   ├── io/           # Data loading
│   ├── preprocessing/# Preprocessing
│   ├── qc/           # Quality control
│   ├── features/     # Feature engineering
│   ├── ml/           # ML/DL integration
│   └── visualization/# Visualization
├── tests/            # Unit tests
├── examples/         # Jupyter notebooks
└── docs/             # Documentation
```

## Adding New Features

1. **Discuss first**: Open an issue to discuss the feature
2. **Branch**: Create a feature branch
3. **Implement**: Write code with tests
4. **Document**: Add docstrings and update docs
5. **Test**: Ensure all tests pass
6. **PR**: Submit a pull request

## Plugin System

PyRamEx supports plugins for extensibility. See `docs/plugins.md` for details.

### Creating a Plugin

```python
from pyramex.plugins import BasePlugin

class MyCustomPreprocessor(BasePlugin):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def process(self, ramanome):
        # Your implementation
        return ramanome
```

## Questions?

- Open a discussion on GitHub
- Contact maintainers: xiaolongxia@openclaw.cn

Thank you for contributing! 🎉
