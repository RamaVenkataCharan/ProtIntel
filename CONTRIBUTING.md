# Contributing to ProtIntel

Thanks for your interest in improving ProtIntel. Contributions from researchers, developers, and users are welcome.

---

## Ways to contribute

You can contribute in several ways:

- fix bugs or improve model logic
- improve dashboard or API behavior
- add tests for model and backend reliability
- improve documentation and examples
- propose new XAI features or visualization enhancements
- optimize training or inference speed

---

## Development setup

1. Fork the repository
2. Clone your fork
3. Create a virtual environment:

```bash
git clone https://github.com/<your-username>/ProtIntel.git
cd ProtIntel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Install frontend dependencies if working on the UI:

```bash
cd frontend
npm install
```

---

## Branching workflow

Use a focused branch for each task:

```bash
git checkout -b feature/my-improvement
```

Use descriptive names such as:

- `fix/sequence-parser-bug`
- `feat/xai-heatmap-improvement`
- `docs/readme-overhaul`

---

## Coding standards

- Write clear, readable code with meaningful names
- Keep logic modular and maintainable
- Follow existing project conventions for Python and frontend code
- Add or update tests when changing behavior
- Keep commits focused and descriptive

---

## Testing

Before submitting a pull request, run the relevant checks:

Python backend / model checks:

```bash
pytest -q
```

Frontend checks:

```bash
cd frontend
npm run build
```

If you are changing model behavior, add tests around the relevant component or function.

---

## Pull request guidelines

When opening a PR:

- explain the problem and solution clearly
- include screenshots for UI changes when relevant
- mention any datasets, config changes, or environment assumptions
- keep the scope focused on one task
- ensure the code passes the relevant tests

---

## Issue reporting

Please open an issue with:

- a clear title
- steps to reproduce
- expected vs actual behavior
- relevant environment details
- logs or screenshots when possible

---

## Code of respect

Please keep discussions professional, constructive, and respectful. We aim to build a collaborative and welcoming project environment.

---

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

---

Thank you for helping make ProtIntel better.
