---
inclusion: always
---

# Python Environment

Always use the virtual environment Python interpreter instead of standalone Python commands.

## Rules

- Use `.venv/bin/python3` instead of `python`, `python3`, or `python3.x`
- Use `.venv/bin/pip` instead of `pip` or `pip3`
- When running Python scripts, tests, or any Python commands, always prefix with `.venv/bin/python3`
- When installing packages, use `.venv/bin/pip install`

## Examples

Good:
```bash
.venv/bin/python3 script.py
.venv/bin/python3 -m pytest tests/
.venv/bin/pip install requests
```

Bad:
```bash
python script.py
python3 -m pytest tests/
pip install requests
```
