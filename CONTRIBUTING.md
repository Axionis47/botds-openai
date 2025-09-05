# Contributing to Bot Data Scientist

Thank you for your interest in contributing to the Bot Data Scientist project! This document provides guidelines for contributing to this OpenAI-led ML pipeline system.

## 🎯 Project Overview

This is a production-ready bot data scientist that uses **OpenAI as the sole decision authority** for all critical ML decisions. The system is designed to be:

- **Deterministic**: Same inputs produce same outputs
- **Traceable**: Complete audit trail of all decisions
- **Professional**: Business-ready reports and documentation
- **Robust**: Budget management and graceful degradation

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API key (required for testing)
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Axionis47/botds-openai.git
cd botds-openai

# Install dependencies
pip install -r requirements.txt
pip install pytest  # For testing

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Test the system
python test_system.py
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_cross_dataset.py -v
python -m pytest tests/test_cache_and_invalidation.py -v
python -m pytest tests/test_acceptance_suite.py -v

# Quick system test
python test_system.py
```

### Test Requirements

All contributions must:
- ✅ Pass existing test suite
- ✅ Include tests for new functionality
- ✅ Maintain OpenAI authority policy
- ✅ Preserve deterministic behavior
- ✅ Follow JSON schema validation

## 📋 Contribution Guidelines

### Code Style

- **Type hints**: All functions must have type hints
- **Docstrings**: All public functions need docstrings
- **Error handling**: Graceful error handling with clear messages
- **Logging**: Use decision log for critical choices

### Architecture Principles

1. **OpenAI Authority**: All critical decisions must go through OpenAI
2. **No Fallbacks**: System must fail fast without OpenAI API key
3. **Handoff Traceability**: All stage outputs must be logged
4. **Schema Validation**: All handoffs must match JSON schemas
5. **Budget Awareness**: All operations must respect resource limits

### Adding New Tools

When adding new tools for OpenAI function calling:

1. **Create the tool class** in `botds/tools/`
2. **Add function definitions** for OpenAI integration
3. **Update tool imports** in `botds/tools/__init__.py`
4. **Add to pipeline** if needed
5. **Write tests** for the new functionality
6. **Update documentation**

Example tool structure:
```python
class NewTool:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
    
    def new_function(self, param: str) -> Dict[str, Any]:
        """Tool function with clear docstring."""
        # Implementation
        return {"result": "value"}
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """OpenAI function calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "NewTool_new_function",
                    "description": "Clear description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param": {"type": "string", "description": "Parameter description"}
                        },
                        "required": ["param"]
                    }
                }
            }
        ]
```

### Adding New Datasets

To add support for new datasets:

1. **Create config file** in `configs/`
2. **Add to DataStore** if needed
3. **Write tests** in `tests/test_cross_dataset.py`
4. **Update documentation**

### Modifying Pipeline Stages

Pipeline changes require special care:

1. **Maintain 7-stage structure**
2. **Preserve OpenAI decision points**
3. **Update handoff schemas** if needed
4. **Test with all three datasets**
5. **Update decision log validation**

## 🔄 Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following the guidelines above
4. **Write/update tests** for your changes
5. **Run the full test suite**: `python -m pytest tests/ -v`
6. **Test with real data**: `python test_system.py`
7. **Update documentation** if needed
8. **Commit with clear messages**
9. **Push to your fork**
10. **Create a Pull Request**

### PR Requirements

- ✅ All tests pass
- ✅ Code follows style guidelines
- ✅ OpenAI authority policy maintained
- ✅ Documentation updated
- ✅ No breaking changes (unless discussed)

## 🐛 Bug Reports

When reporting bugs, please include:

- **System information**: OS, Python version
- **Error message**: Full traceback
- **Configuration**: Anonymized config file
- **Steps to reproduce**: Minimal example
- **Expected vs actual behavior**

## 💡 Feature Requests

For new features, please:

- **Check existing issues** first
- **Describe the use case** clearly
- **Consider OpenAI integration** requirements
- **Think about budget impact**
- **Propose implementation approach**

## 🔒 Security

- **Never commit API keys** or sensitive data
- **Use .env files** for local development
- **Report security issues** privately
- **Follow responsible disclosure**

## 📚 Documentation

Help improve documentation by:

- **Fixing typos** and unclear explanations
- **Adding examples** for complex features
- **Updating runbook** with new troubleshooting
- **Improving code comments**

## 🎖️ Recognition

Contributors will be recognized in:

- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Code Review**: For implementation guidance

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make the Bot Data Scientist better! 🚀
