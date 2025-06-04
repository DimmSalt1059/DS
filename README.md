# DS

This repository demonstrates a simple "cell-based" language model designed to run on low-power hardware. The examples avoid external machine-learning libraries and rely only on the Python standard library.

## cells.py

`cells.py` implements a miniature neural network using multiple *cells*. Each cell performs a linear transformation followed by a tanh activation. The `CellModel` class chains several cells together. A small training loop is included to learn an identity mapping over a tiny alphabet.

Run the script with:

```bash
python3 cells.py
```

It will train the model and then print a test prediction for the character `h`.

## simple_language_model.py

`simple_language_model.py` builds on `cells.py` and demonstrates a minimal pipeline resembling the layered architecture discussed in the repository issues. It includes an encoder, a cognitive core composed of `Cell` objects, and a lightweight API layer.

Run the script with:

```bash
python3 simple_language_model.py
```

This will train the model on an identity task and output the predicted character for a test input. The example keeps the code lightweight so it can be experimented with on hardware in the ~20&nbsp;W power range.
