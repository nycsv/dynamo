# Dynamo Development Guide

## Python Package Manager: `uv` (Recommended)

All development and installation workflows should use **`uv`** instead of `pip` directly.

### Why `uv`?

- **10-100× faster** than pip (parallelized downloads, Rust-based implementation)
- **Reliable PEP 660 editable installs** (pip still has issues in some cases)
- **Deterministic builds** via `uv.lock` files
- **Integrated Python version management** (like `pyenv` but built-in)
- **Cross-platform** support (Windows, macOS, Linux)
- **Single binary** — no complex environment setup needed

### Installation

```bash
# macOS
brew install uv

# Linux (x86_64 or aarch64)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

---

## Development Workflow

### Quick Start (Recommended)

```bash
# Install uv (recommended Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv venv
source venv/bin/activate

# Clone repository and install Dynamo in development mode
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo

uv pip install -e ".[vllm]"  # Install with vLLM backend
```

### Step-by-Step Setup

#### 1. Install `uv` (recommended)

```bash
# Install uv (10-100× faster than pip, reliable editable installs)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

#### 2. Clone Repository

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
```

#### 3. Create Virtual Environment

```bash
# Using uv (preferred — faster, cleaner)
uv venv venv
source venv/bin/activate    # Linux / macOS
# or
venv\Scripts\activate       # Windows (PowerShell)
```

#### 4. Install Dynamo in Development Mode

For **base framework**:

```bash
# Install with vLLM backend (default)
uv pip install -e ".[vllm]"

# Or install with TRT-LLM backend
uv pip install -e ".[trtllm]"

# Or install with SGLang backend
uv pip install -e ".[sglang]"
```

For **multimodal ASR examples** (after base installation):

```bash
cd examples/multimodal

# Quick setup (one-time — handles venv, services, model)
bash launch/install.sh

# Or manual with uv:
uv pip install vllm-omni==0.16.0rc1 'vllm[audio]' accelerate cupy-cuda12x
```

#### 5. Verify Installation

```bash
python -c "import dynamo; print('✓ Dynamo installed')"
python -c "import vllm; print('✓ vLLM installed')"
python -c "import torch; print(f'✓ PyTorch with CUDA: {torch.cuda.is_available()}')"
```

---

## Common Development Tasks

### Install Additional Packages

```bash
# Install a single package
uv pip install package-name

# Install multiple packages
uv pip install package1 package2 package3

# Install with extras
uv pip install 'package-name[extra1,extra2]'
```

### Update Packages

```bash
# Upgrade a package
uv pip install --upgrade package-name

# Upgrade all packages
uv pip install --upgrade
```

### View Installed Packages

```bash
# List all packages
uv pip list

# Show details about a package
uv pip show package-name

# Check for outdated packages
uv pip list --outdated
```

### Create Lock Files (for reproducibility)

```bash
# Generate uv.lock from pyproject.toml
uv lock

# Or generate requirements.txt
uv pip compile pyproject.toml -o requirements.txt

# Sync environment to lock file (reproducible installs)
uv pip sync requirements.txt
```

### Remove Packages

```bash
uv pip uninstall package-name
```

---

## IDE Integration

### VS Code

1. Install **Python** extension by Microsoft
2. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
3. Search for "Python: Select Interpreter"
4. Choose the venv interpreter (usually `./venv/bin/python` or `venv\Scripts\python.exe`)

VS Code will auto-detect the venv created by `uv venv` or Python's `venv`.

### PyCharm / IntelliJ

1. Go to **Settings** → **Project: dynamo** → **Python Interpreter**
2. Click the gear ⚙️ icon → **Add...**
3. Select **Existing Environment**
4. Browse to `venv/bin/python` (or `venv\Scripts\python.exe` on Windows)
5. Click **OK**

PyCharm will then recognize the venv and provide full autocomplete/linting.

### Vim / Neovim (with LSP)

Example config for `pyright` LSP:

```lua
-- nvim/init.lua
require("lspconfig").pyright.setup({
  settings = {
    python = {
      pythonPath = vim.fn.expand("~/dynamo/venv/bin/python"),
    }
  }
})
```

Or use `pylance` for more features (requires VS Code).

---

## Testing

### Run Unit Tests

```bash
# Install test dependencies
uv pip install pytest pytest-cov pytest-asyncio

# Run all tests
pytest

# Run specific test file
pytest tests/test_module.py

# Run with coverage
pytest --cov=dynamo tests/

# Run with verbose output
pytest -v
```

### Test Multimodal ASR Pipeline

```bash
cd examples/multimodal

# Launch pipeline (in background or another terminal)
bash launch/asr_speechllm_agg.sh &

# Run validation suite
bash launch/validate_asr_speechllm_agg.sh

# Or run manually with pytest
uv pip install pytest httpx
pytest tests/test_asr_encode_worker.py -v
```

---

## Code Quality

### Linting and Formatting (Optional)

```bash
# Install tools
uv pip install ruff black mypy

# Format code
ruff check --fix .
black examples/

# Type checking
mypy components/processor.py

# View all issues without fixing
ruff check .
```

### Pre-commit Hooks (Optional)

To run checks automatically before each commit:

```bash
# Install pre-commit
uv pip install pre-commit

# Initialize hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

---

## Troubleshooting Development Setup

### Issue: `ModuleNotFoundError: No module named 'dynamo'`

**Solution:** Ensure you've installed Dynamo in editable mode:

```bash
uv pip install -e .
```

### Issue: CUDA not found

**Solution:** Verify NVIDIA toolkit is installed:

```bash
nvcc --version
echo $CUDA_HOME
nvidia-smi
```

### Issue: vllm-omni import fails

**Solution:** For multimodal work, vllm-omni is required:

```bash
uv pip install vllm-omni==0.16.0rc1

# Or build from source:
git clone --depth 1 --branch v0.16.0rc1 https://github.com/vllm-project/vllm-omni.git /tmp/vllm-omni
uv pip install /tmp/vllm-omni
```

### Issue: `uv` command not found

**Solution:** Install uv or add it to PATH:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use Python fallback (slower)
pip install uv
```

---

## References

- [uv Documentation](https://docs.astral.sh/uv/) — comprehensive uv guide
- [Dynamo README](README.md) — project overview
- [Multimodal ASR Tutorial](examples/multimodal/QWEN3_ASR_A100_TUTORIAL.md) — complete A100×8 setup
- [Multimodal README](examples/multimodal/README.md) — quick reference for ASR pipeline
