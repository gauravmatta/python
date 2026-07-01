# Fixes Applied for Python 3.14+ Compatibility

## Issue 1: ast.NameConstant AttributeError
**Problem**: Python 3.14 removed `ast.NameConstant`, `ast.Num`, `ast.Str`, etc., replacing them with `ast.Constant`. Libraries like `docstring_parser` that reference the old AST nodes fail.

**Solution**: Created `/Users/gauravmatta/Projects/python/crewai/.venv/lib/python3.14/site-packages/sitecustomize.py` which automatically patches the `ast` module when Python starts, mapping legacy names to `ast.Constant`.

## Issue 2: ModuleNotFoundError: No module named 'pkg_resources'
**Problem**: `setuptools` was installed but `pkg_resources` module wasn't available. This is a Python 3.14 compatibility issue.

**Solution**: Created a minimal `/Users/gauravmatta/Projects/python/crewai/.venv/lib/python3.14/site-packages/pkg_resources.py` that provides the necessary functionality using `importlib.metadata` instead.

## Issue 3: Code ordering in basic.py
**Problem**: The compatibility shim in `basic.py` was placed after the `streamlit` import, which was too late.

**Solution**: Moved the compatibility shim to the very beginning of `basic.py` before all other imports.

## Testing
All fixes have been verified:
- ✓ CrewAI imports successfully
- ✓ The basic.py script loads without errors
- ✓ Streamlit can run the app without import errors

You can now run: `streamlit run basic.py`

