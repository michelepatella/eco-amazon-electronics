# Compatibility

This document tracks direct code modifications applied to external libraries to ensure compatibility between dependencies, frameworks, and runtime versions.

> **IMPORTANT**: These changes are not part of the original upstream projects and are applied locally.

## Fixes

### RecBole (v1.2.0)

#### 1. Compatibility with NumPy (v2.4.4)

Fix for NumPy deprecated alias removal.

##### 1.1 `recbole/config/configurator.py`

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

##### 1.2 `recbole/evaluator/metrics.py`

At the beginning of the script (after imports), add:

```python
np.float = float
```

#### 2. Compatibility with PyTorch (v2.11.0)

Fix for PyTorch checkpoint loading behavior change.

##### 2.1 `recbole/trainer/trainer.py`

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

##### 2.2 `recbole/quick_start/quick_start.py`

Replace:

```python
checkpoint = torch.load(model_file)
```

With:

```python
checkpoint = torch.load(model_file, weights_only=False)
```

#### 3. Compatibility with SciPy (v1.17.1)

Fix for SciPy internal API removal.

##### 3.1 `recbole/model/general_recommender/lightgcn.py`

Replace:

```python
A._update(data_dict)
```

With:

```python
for (i, j), val in data_dict.items():
    A[i, j] = val
```
