# Reto's toolbox

Here is the documentation for Reto's toolbox.
Well, this is most likely for myself
and thus mostly consists of an automatically generated
API documentation.

On the right-hand side you can find
the available tools. 

## Installation

To install the tools into your environment, 
you should install from `git`.

```
pip install git+https://github.com/trappitsch/rttools.git
```


## Units

Note that [`pint`](https://pint.readthedocs.io/en/stable/) 
is used to provide unitful quantities to certain routiens.
These are indicated by the `Quantity` annotation.

To use units, the following gives an example:

```python
from rttools import ureg  # the unit registry

voltage = 3 * ureg.V
current = 1.5 * ureg.A

resistance = voltage / current
```

In this example, `resistance` will now be unitful.
This is very helpful for avoiding unit conversion mistakes!

!!! note "Unitful array-like objects"

    To create unitful array-like objects, 
    please assign a unit to the whole array and NOT to the individual arrays.

    ```python
    import numpy as np

    from rttools import ureg

    currents = np.array([1.2, 3.5, 17.9]) * ureg.A
    ```
