import yaml

from .spec import FIELD_DESCRIPTIONS


class InvalidConfigException(Exception):
    """Raised for general config errors"""

    pass


class InvalidConfigVariableException(Exception):
    """Raised when a config variable is present but is not part of the config spec"""

    pass


class MissingConfigVariableException(Exception):
    """Raised when a config variable is missing from the config file that is required"""

    pass


def _get_nested_from_spec(data: dict = FIELD_DESCRIPTIONS, path=[]):
    """
    Get a nested value from a dictionary using a list of keys
    allowing for a nested structure to be traversed. A special value named `_values_` is used to indicate
    that any key can be used at this level of the dictionary.
    """
    key = path[0]
    if key in data:
        children = data[key]
    elif "_values_" in data:
        children = data["_values_"]
    else:
        raise KeyError(f"Key {key} not found in {data}")

    if len(path) == 1:
        return children
    else:
        return _get_nested_from_spec(children, path=path[1:])


class ConfigDict(dict):
    """
    Configuration holder that behaves like a dictionary with two differences:

    1. All key-fetches are checked against the configuration specification to
       check whether a specific configuration variable is part of the spec.
       (and to provide inline documentation when a variable in the spec
       hasn't been set in the config)
    2. All nested dictionaries are wrapped in `ConfigDict` with a reference to
       the parent in the config hierarchy which enables indexing into the right
       part of the spec

    """

    def __init__(self, config_dict, parents=[]):
        super().__init__(config_dict)

        self._parents = parents
        for k, v in self.items():
            item_parents = parents + [k]

            if isinstance(v, dict):
                self[k] = ConfigDict(v, parents=item_parents)

    def __getitem__(self, __key):
        found_value = False
        try:
            value = super().__getitem__(__key)
            found_value = True
        except KeyError:
            pass

        field_path = self._parents + [__key]
        path_joined = ".".join(field_path)
        try:
            field_desc = _get_nested_from_spec(path=field_path)
            if "_doc_" in field_desc:
                field_desc = field_desc["_doc_"]

            if found_value:
                return value

        except KeyError as ex:
            if found_value:
                raise InvalidConfigVariableException(
                    f"Although a value for {path_joined} does exist ({value})"
                    " this parameter shouldn't be included in the configuration"
                    " as it is not part of the config spec."
                )
            else:
                raise KeyError(f"Unknown config variable `{path_joined}`") from ex

        raise MissingConfigVariableException(
            f"Missing config variable `{path_joined}` ({field_desc})"
        )

    def get(self, key, default=None):
        try:
            return self[key]
        except MissingConfigVariableException:
            return default

    @classmethod
    def load(cls, fp_config):
        with open(fp_config, "r") as fh:
            config_dict = yaml.load(fh, Loader=yaml.FullLoader)
            return cls(config_dict)
