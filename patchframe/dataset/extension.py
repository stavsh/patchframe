
from copy import deepcopy
import pandas as pd
import weakref

_FIELD = "_patchframe_field"

@pd.api.extensions.register_series_accessor("finfo")
class FieldAccessor:
    def __init__(self, obj: pd.Series):
        self._obj = obj

    @property
    def field(self):
        return self._obj.attrs.get(_FIELD)

    @field.setter
    def field(self, value):
        self._obj.attrs[_FIELD] = value
