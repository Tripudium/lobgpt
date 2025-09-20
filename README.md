# LOB-GPT
Transformer for Limit Order Books

## Setup

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lobgpt
```

2. Install dependencies using uv:
```bash
uv sync
```

This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

3. Configure environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add your API keys:
- `TARDIS_API_KEY`: Get from [tardis.dev](https://tardis.dev/)
- `BYBIT_API_KEY` and `BYBIT_API_SECRET`: Get from [Bybit API Management](https://www.bybit.com/app/user/api-management)

### Development

To install the package in editable mode for development:
```bash
uv pip install -e .
```

### Usage

- Import modules in Python:
```python
from lobgpt.hdb import get_dataset

dl = get_dataset("tardis")
df = dl.load_book("BTCUSDT", ['250912.000100', '250913.215000'], depth=10)
```
