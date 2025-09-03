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
    from baltic_bokeh import plotRectangularTree
    from baltic_bokeh import plotRectangularPoints
    from baltic_bokeh import plotCircularTree
    from baltic_bokeh import plotCircularPoints
    from baltic_bokeh import addText

    # Verify they are callable
    assert callable(plotRectangularTree)
    assert callable(plotRectangularPoints)
    assert callable(plotCircularTree)
    assert callable(plotCircularPoints)
    assert callable(addText)


def test_module_docstring():
    """Test that the module has proper documentation."""
    import baltic_bokeh

    assert baltic_bokeh.__doc__ is not None
    assert "Baltic-Bokeh" in baltic_bokeh.__doc__
    assert "interactive" in baltic_bokeh.__doc__.lower()


if __name__ == "__main__":
    test_package_structure()
    test_function_imports()
    test_module_docstring()
    print("All import tests passed!")
