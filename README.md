## Workflow
```text
                 ┌─────────────────────────────────────────────────────────┐
                 │                    Amazon Reviews'23                    │
                 │                                                         │
                 │  ┌─────────────────────┐       ┌─────────────────────┐  │
                 │  │    User Reviews     │       │    Item Metadata    │  │
                 │  └──────────┬──────────┘       └──────────┬──────────┘  │
                 └─────────────┼─────────────────────────────┼─────────────┘
                               │                             │
                       ┌───────▼───────┐             ┌───────▼───────┐
                       │ RecSys Model  │             │ PCF Enricher  │
                       └───────┬───────┘             └───────┬───────┘
                               │                             │
                       ┌───────▼───────┐             ┌───────▼───────┐
                       │     Model     │             │   Enriched    │
                       │  Predictions  │             │ Item Metadata │
                       └───────┬───────┘             └───────┬───────┘
                               │                             │
                               └──────────────┬──────────────┘
                                              │
                                      ┌───────▼───────┐
                                      │  Re-Ranking   │
                                      │    Module     │
                                      └───────┬───────┘
                                              │
                                      ┌───────▼───────┐
                                      │  Sustainable  │
                                      │  Predictions  │
                                      └───────────────┘
```

## RecBole (v1.2.0) Compatibility Fixes

### 1. `recbole/config/configurator.py`

> Fix for NumPy compatibility issue due to removal of deprecated aliases in recent versions.

Replace:

```python
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.str = np.str_
np.long = np.int_
np.unicode = np.unicode_
```

With:

```python
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.long = int
np.unicode = str
```

### 2. `recbole/trainer/trainer.py`

> Fix for PyTorch checkpoint loading to ensure consistent model deserialization across CPU/GPU environments and compatibility with newer PyTorch versions.

Replace:

```python
checkpoint = torch.load(checkpoint_file, map_location=self.device)
```

With:

```python
checkpoint = torch.load(
    checkpoint_file,
    map_location=self.device,
    weights_only=False
)
```

### 3. `recbole/model/general_recommender/lightgcn.py`

> Fix for deprecated sparse matrix method `_update()` removed in modern SciPy versions.

Replace:

```python
A._update(data_dict)
```

With:

```python
for (i, j), val in data_dict.items():
    A[i, j] = val
```
