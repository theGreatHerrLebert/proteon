# proteon-connector

PyO3 bindings layer for the [proteon](https://github.com/theGreatHerrLebert/proteon)
structural bioinformatics toolkit.

This package is the compiled Rust ↔ Python bridge. Most users should install the
`proteon` Pythonic wrapper instead:

```bash
pip install proteon
```

`proteon-connector` is pulled in automatically as a dependency and exposes the raw
`#[pyclass]` types the wrapper composes on top of. Direct use is supported but not
the recommended entry point — see the main repository for documentation, examples,
and the public API.

## Project

- Source: https://github.com/theGreatHerrLebert/proteon
- License: MIT
