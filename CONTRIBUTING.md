# Contributing to VisionCanvas

Thank you for your interest in contributing to **VisionCanvas**! Whether you are fixing a bug, adding new gesture capabilities, optimizing algorithms, or improving documentation, your contributions are welcome.

Please take a moment to review these contribution guidelines to ensure a smooth collaboration.

---

## Getting Started

1. **Fork the Repository:**  
   Fork the repository on GitHub to your personal account:  
   `https://github.com/srushtijayaraman/VisionCanvas/fork`

2. **Clone Your Fork:**  
   ```bash
   git clone https://github.com/<your-username>/VisionCanvas.git
   cd VisionCanvas
   ```

3. **Set Up a Clean Virtual Environment:**  
   ```bash
   python -m venv .venv

   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1

   # Windows (cmd):
   .\.venv\Scripts\activate.bat

   # Linux/macOS:
   source .venv/bin/activate
   ```

4. **Install Dependencies:**  
   Install both runtime and test dependencies:  
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

5. **Verify the Test Suite Baseline:**  
   Ensure all existing tests pass before making any changes:  
   ```bash
   pytest -v
   ```

---

## Development Guidelines

- **Branch Naming:** Create a feature or bugfix branch with a descriptive name:  
  `git checkout -b feat/my-new-feature` or `git checkout -b fix/issue-description`
- **Code Style:**  
  - Write readable, idiomatic Python (Python 3.10+).
  - Include clear docstrings for all new classes and methods.
  - Keep computer-vision algorithms explainable, modular, and mathematically sound.
  - Do not introduce heavy third-party dependencies unless strictly necessary and discussed beforehand.
- **Hardware Agility:**  
  - Ensure any new camera or rendering features handle camera disconnection or missing frames gracefully without crashing.
  - Avoid blocking operations in the primary render loop.

---

## Testing Your Changes

All contributions must include appropriate automated tests or verification evidence:

1. **Add Unit / Integration Tests:**  
   If you add a new feature or fix a bug, add corresponding test cases under the `tests/` directory following the naming pattern `tests/test_<module>.py`.
2. **Run Full Test Suite:**  
   ```bash
   pytest -v
   ```
   All tests must pass with zero failures.
3. **Verify Environment Consistency:**  
   ```bash
   python -m pip check
   ```

---

## Submitting a Pull Request

1. **Commit Your Changes:**  
   Write clear, concise commit messages following standard conventions (e.g., `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
2. **Push to Your Fork:**  
   ```bash
   git push -u origin feat/my-new-feature
   ```
3. **Open a Pull Request:**  
   - Navigate to the repository on GitHub and click **New Pull Request**.
   - Provide a clear title and description explaining what was changed, the motivation, and test evidence demonstrating that the change works.
   - Mention any related issues if applicable.

---

## Reporting Issues & Feature Suggestions

- **Bug Reports:** If you find a bug, please open an issue using the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) with steps to reproduce, expected vs. actual behavior, and environment details.
- **Feature Requests:** Have an idea for a new brush, gesture, or interface enhancement? Open an issue using the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) describing the proposal and use case.
