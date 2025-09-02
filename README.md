# Baltic-Bokeh

[![Tests](https://github.com/sanjaynagi/baltic-bokeh/workflows/Tests/badge.svg)](https://github.com/sanjaynagi/baltic-bokeh/actions)
[![CI](https://github.com/sanjaynagi/baltic-bokeh/workflows/CI/badge.svg)](https://github.com/sanjaynagi/baltic-bokeh/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Interactive phylogenetic tree visualization using Bokeh and Baltic.

This package provides interactive Bokeh-based plotting functionality for Baltic phylogenetic trees, offering enhanced interactivity and metadata integration support compared to static matplotlib plots.

## Installation

```bash
pip install baltic-bokeh
```

## Usage

```python
import baltic as bt
import baltic_bokeh
import pandas as pd

# Load tree and metadata
tree = bt.loadNewick('tree.newick')
metadata = pd.read_csv('metadata.tsv', sep='\t', index_col=0)

# Create interactive circular tree plot
p = baltic_bokeh.plotCircularTree(tree, plot_width=800, plot_height=800)

# Add interactive points with metadata coloring
p = baltic_bokeh.plotCircularPoints(
    tree, 
    p=p,
    df_metadata=metadata,
    color_column='species',
    hover_data=['location', 'date']
)

# Show plot
from bokeh.plotting import show
show(p)
```

## Features

- Interactive phylogenetic tree visualization
- Circular and rectangular tree layouts  
- Metadata integration with color mapping
- Hover tooltips with sample information
- Bokeh-powered interactivity (zoom, pan, etc.)

## Dependencies

- baltic >= 0.3.0
- bokeh >= 3.0.0
- pandas >= 1.0.0
- numpy >= 1.18.0

## Development

### Installation for Development

```bash
git clone https://github.com/sanjaynagi/baltic-bokeh.git
cd baltic-bokeh
pip install -e .
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run tests with coverage
pytest --cov=baltic_bokeh

# Run specific test files
pytest tests/test_bokeh_plotting.py -v
```

### Example Usage

See the `examples/` directory for complete working examples:

```bash
cd examples/
python example.py
```