# Contributing to PyMC MMM Workbench

Thank you for your interest in contributing! This project welcomes contributions from the community.

## Getting Started

### Prerequisites
- Python 3.11+
- `pip` and `venv`

### Development Setup

```bash
# Clone the repository
git clone https://github.com/hrczggyrgy/pymc_MMM_workbench.git
cd pymc_MMM_workbench

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest ruff

# Run the app locally
streamlit run app.py
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=utils --cov-report=html

# Run linting
ruff check .
```

## Code Style

- Follow PEP 8 (enforced by `ruff`)
- Type hints encouraged for public functions
- Docstrings for public functions (NumPy style)
- Max line length: 88 characters (ruff default)

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Make your changes
3. Run tests and linting: `pytest tests/ && ruff check .`
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to your fork (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring
- `docs:` documentation changes
- `test:` test additions/modifications
- `chore:` maintenance tasks

Example: `feat: add st.status() for model fitting progress`

## Reporting Issues

Use the GitHub issue templates:
- Bug report: describe the bug, steps to reproduce, expected vs actual behavior
- Feature request: describe the feature, use case, and any alternatives considered

## Code of Conduct

Be respectful and inclusive. Follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).