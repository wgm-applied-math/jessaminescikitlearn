---
---

# Handling parameters to an algorithm

According to [the developer documentation](https://scikit-learn.org/stable/developers/develop.html):

The `__init__` method for a regressor should take a bunch of arguments, all of which should be stored unmodified as attributes with exactly the same name on `self`.

The `BaseEstimator.get_params` function uses python magic to go through the definition of `__init__` and fish out these parameters, which are pulled from attributes of `self` into a `dict`.

Validation and normalization of parameters is confined to the `fit` method, and must not be done in `__init__`.

Any _statistical_ parameter that is computed from data should result in an attribute set in `fit` and the convention is that its name ends in one underscore:

```python
self.mean_ = mean(...)
```

The function `validate_parameter_constraints`, which is called by `check_estimator` and is available through `BaseEstimator._validate_params`, looks for an object's `_parameter_constraints` attribute and checks them out.
The documentation for `validate_parameter_constraints` has the details, but basically it should be a `dict` with parameter names as keys, each mapped to a list of constraints, and if any of them are satisfied by a particular value, the value is accepted.
So it's a logical or, and you can't have just one constraint.

The `check_is_fitted` function by default returns `True` if it finds at least one attribute of this form on an estimator, that is, it begins with a non-underscore and ends in one underscore.

The function `sklearn.utils.validation.validate_data` checks that the arrays or tables are well formed, and outputs plain arrays for `X` and `y`.
I think that the idea is to call this with `reset=True` in `fit` to store column names, and with `reset=False` in `predict` to validate that the input table has the expected columns matching the fit data.

The function `check_estimator` does a bunch of tests on an estimator, not just to see that everything is implemented and named correctly, but that it handles a variety of pre-built test cases.
