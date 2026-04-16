# 🤝 Contributing to MLens

Thank you for considering contributing to **MLens**! This project thrives on community input — whether it's a bug fix, new feature, documentation improvement, or a fresh idea.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

---

## 📜 Code of Conduct

This project follows a simple rule: **be kind and respectful**. All contributors are expected to maintain a welcoming environment regardless of experience level, background, or identity.

Unacceptable behaviour can be reported to the maintainers directly via GitHub Issues.

---

## 🌟 Ways to Contribute

You don't have to write code to contribute! Here's how you can help:

| Type | Examples |
|---|---|
| 🐛 **Bug Reports** | Found something broken? Open an issue |
| ✨ **New Features** | PyTorch support, PDF reports, CLI tool |
| 📖 **Documentation** | Fix typos, improve examples, add docstrings |
| 🧪 **Tests** | Add missing test cases, improve coverage |
| 🎨 **Visuals** | Better charts, dashboard UI improvements |
| 💬 **Discussion** | Answer questions in Issues, share ideas |

---

## 🚀 Getting Started

### 1. Fork the Repository
Click **"Fork"** on the top right of the MLens GitHub page.

### 2. Clone Your Fork
```bash
git clone https://github.com/saiganesh47/mlens.git
cd mlens
```

### 3. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e ".[dev]"         # installs dev extras (pytest, black, flake8)
```

### 5. Verify Setup
```bash
pytest tests/ -v
```
All tests should pass before you start making changes.

---

## 🔄 Development Workflow

### 1. Create a Branch
Always work on a new branch — never directly on `main`:
```bash
# For a new feature
git checkout -b feat/pytorch-support

# For a bug fix
git checkout -b fix/shap-kernel-explainer-crash

# For docs
git checkout -b docs/improve-quickstart
```

### 2. Make Your Changes
Write clean, documented code (see [Coding Standards](#coding-standards)).

### 3. Run Tests
```bash
pytest tests/ -v --cov=mlens --cov-report=term-missing
```

### 4. Push & Open a PR
```bash
git add .
git commit -m "✨ feat: add native PyTorch support to ShapAnalyzer"
git push origin feat/pytorch-support
```
Then open a **Pull Request** on GitHub against the `main` branch.

---

## ✍️ Commit Message Guidelines

We use **semantic commit messages** for a clean, readable history:

| Prefix | When to use |
|---|---|
| `✨ feat:` | New feature |
| `🐛 fix:` | Bug fix |
| `📖 docs:` | Documentation only |
| `🧪 test:` | Adding or fixing tests |
| `♻️ refactor:` | Code refactor, no behaviour change |
| `⚡ perf:` | Performance improvement |
| `🔧 chore:` | Build, CI, config changes |

**Examples:**
```
✨ feat: add intersectional fairness evaluation
🐛 fix: handle constant features in PSI calculation
📖 docs: add PyTorch model example to README
🧪 test: add edge cases for KernelExplainer fallback
```

---

## 🔍 Pull Request Process

1. **One PR per feature/fix** — keep it focused
2. **Fill in the PR template** — describe what changed and why
3. **Link related issues** — use `Closes #42` in your PR description
4. **All tests must pass** — CI will run automatically
5. **Request a review** — tag a maintainer if no one responds in 3 days
6. **Squash on merge** — we keep a clean commit history

### PR Title Format
```
✨ feat: Add PyTorch support to ShapAnalyzer
🐛 fix: Resolve KS-test crash on constant features
📖 docs: Add fairness metrics explanation to README
```

---

## 🧹 Coding Standards

### Style
- Follow **PEP 8** — use `black` for auto-formatting:
```bash
black mlens/ tests/
```

### Type Hints
- All public functions must have **type annotations**:
```python
# ✅ Good
def explain(self, X_test: np.ndarray) -> ShapResult:

# ❌ Bad
def explain(self, X_test):
```

### Docstrings
- Use **NumPy-style docstrings** for all public classes and methods:
```python
def top_features(self, n: int = 10) -> List[Dict[str, Any]]:
    """
    Return the top-n most important features sorted by mean |SHAP|.

    Parameters
    ----------
    n : int
        Number of top features to return (default: 10).

    Returns
    -------
    list of dict
        Each dict contains: 'rank', 'name', 'mean_abs_shap'.
    """
```

### Tests
- Every new feature needs a corresponding test in `tests/`
- Use `pytest` fixtures for shared setup
- Aim for **>80% coverage** on new code

---

## 🐛 Reporting Bugs

Open a GitHub Issue with the following template:

```
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behaviour:
1. Model type used (e.g. XGBClassifier)
2. Code snippet that causes the error
3. Full error traceback

**Expected behaviour**
What you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 22.04]
- Python version: [e.g. 3.10.4]
- MLens version: [e.g. 0.1.0]
- Key library versions: shap, fairlearn, scipy
```

---

## 💡 Suggesting Features

Open a GitHub Issue with the label `enhancement` and describe:

1. **The problem** you're trying to solve
2. **Your proposed solution**
3. **Alternatives you considered**
4. **Any relevant references** (papers, libraries, examples)

---

## 🗺️ Roadmap (Good First Issues)

Looking for somewhere to start? These are high-priority and beginner-friendly:

- [ ] `🔧` Add a CLI: `mlens audit model.pkl X_test.csv`
- [ ] `🧪` Increase test coverage to 90%+
- [ ] `📖` Add Jupyter Notebook tutorial
- [ ] `✨` PDF report export via `weasyprint`
- [ ] `✨` Concept drift detection (ADWIN algorithm)
- [ ] `✨` Intersectional fairness (multi-attribute analysis)

Issues tagged **`good first issue`** are great starting points!

---

## 🙏 Thank You

Every contribution — no matter how small — makes MLens better for the entire ML community. We appreciate your time and effort!

**Happy auditing! 🔬**
