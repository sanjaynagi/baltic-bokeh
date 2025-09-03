"""
Simple import tests that don't require external dependencies.
"""

import sys
import os

# Add the parent directory to Python path so we can import baltic_bokeh
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_package_structure():
    """Test that the package structure is correct."""
    import baltic_bokeh

    # Test that main module attributes exist
    assert hasattr(baltic_bokeh, "__version__")
    assert hasattr(baltic_bokeh, "__all__")

    # Test that expected functions are in __all__
    expected_functions = [
        "plot_bokeh_scatter",
        "plotTree",
        "plotRectangularTree",
        "plotRectangularPoints",
        "plotCircularTree",
        "plotCircularPoints",
        "addText",
    ]

    assert set(baltic_bokeh.__all__) == set(expected_functions)


def test_function_imports():
    """Test that all functions can be imported."""
    import baltic_bokeh

    # Test individual function imports
    from baltic_bokeh import plot_bokeh_scatter
    from baltic_bokeh import plotTree
    from baltic_bokeh import plotRectangularTree
    from baltic_bokeh import plotRectangularPoints
    from baltic_bokeh import plotCircularTree
    from baltic_bokeh import plotCircularPoints
    from baltic_bokeh import addText

    # Verify they are callable
    assert callable(plot_bokeh_scatter)
    assert callable(plotTree)
    assert callable(plotRectangularTree)
    assert callable(plotRectangularPoints)
    assert callable(plotCircularTree)
    assert callable(plotCircularPoints)
    assert callable(addText)


if __name__ == "__main__":
    test_package_structure()
    test_function_imports()
    print("All import tests passed!")
