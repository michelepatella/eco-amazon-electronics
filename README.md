## Workflow
```text
                  ┌─────────────────────────────────────────────────────────┐
                  │                    Amazon Reviews'23                    │
                  │                                                         │
                  │  ┌---------------------┐       ┌---------------------┐  │
                  │  │    User Reviews     │       │    Item Metadata    │  │
                  │  └----------┬----------┘       └----------┬----------┘  │
                  └─────────────┼─────────────────────────────┼─────────────┘
                                │                             │
                     ╔══════════▼══════════╗       ╔══════════▼══════════╗
                     ║  Data Preprocessing ║       ║ Emission Enrichment ║
                     ╚══════════┬══════════╝       ╚══════════┬══════════╝
                                │                             │
                     ╔══════════▼══════════╗       ┌----------▼----------┐
                     ║    Model Training   ║       │  Emission-Enriched  │
                     ╚══════════┬══════════╝       │    Item Metadata    │
                                │                  └----------┬----------┘
                     ╔══════════▼══════════╗                  │
                     ║   Model Inference   ║                  │
                     ╚══════════┬══════════╝                  │
                                │                             │
                     ┌----------▼----------┐                  │
                     │   Recommendations   │                  │
                     └----------┬----------┘                  │
                                │                             │
                                └──────────────┬──────────────┘
                                               │
                                ╔══════════════▼══════════════╗
                                ║     Sustainability-aware    ║
                                ║  Recommendation Re-ranking  ║
                                ╚══════════════┬══════════════╝
                                               │
                                ┌--------------▼--------------┐
                                │  Re-ranked Recommendations  │
                                └--------------┬--------------┘
                                               │
                                ╔══════════════▼══════════════╗
                                ║       Model Evaluation      ║
                                ╚═════════════════════════════╝
```

## RecBole (v1.2.0) Compatibility Fixes

### 1. `recbole/config/configurator.py`, `recbole/evaluator/metrics.py`

> Fix for NumPy compatibility issue due to removal of deprecated aliases in recent versions.


#### 1.1 `recbole/config/configurator.py`

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

#### 1.2 `recbole/evaluator/metrics.py`

At the beginning of the script, add:

```python
np.float = float
```

### 2. `recbole/trainer/trainer.py`, `recbole/quick_start/quick_start.py`

> Fix for PyTorch checkpoint loading to ensure consistent model deserialization across CPU/GPU environments and compatibility with newer PyTorch versions.

#### 2.1 `recbole/trainer/trainer.py`
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

#### 2.2 `recbole/quick_start/quick_start.py`
Replace:

```python
checkpoint = torch.load(model_file)
```

With:

```python
checkpoint = torch.load(model_file, weights_only=False)
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
