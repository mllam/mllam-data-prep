import yaml

from .spec import FIELD_DESCRIPTIONS


class InvalidConfig(Exception):
    pass


def _get_nested_from_spec(data: dict = FIELD_DESCRIPTIONS, path=[]):
    """
    Get a nested value from a dictionary using a list of keys
    allowing for a nested structure to be traversed. A special value named `_values_` is used to indicate
    that any key can be used at this level of the dictionary.
    """
    if len(path) == 1:
        return data[path[0]]
    else:
        key = path[0]
        if key in data:
            children = data[key]
        elif "_values_" in data:
            children = data["_values_"]
        else:
            raise KeyError(f"Key {key} not found in {data}")

        return _get_nested_from_spec(children, path=path[1:])


class Config(dict):
    def __init__(self, config_dict, parents=[]):
        super().__init__(config_dict)

        self._parents = parents
        for k, v in self.items():
            item_parents = parents + [k]

            if isinstance(v, dict):
                self[k] = Config(v, parents=item_parents)

    def __getitem__(self, __key):
        try:
            return super().__getitem__(__key)
        except KeyError:
            pass

        field_path = self._parents + [__key]
        path_joined = ".".join(self._parents)
        try:
            field_desc = _get_nested_from_spec(path=field_path)
        except KeyError as ex:
            raise KeyError(
                f"Unknown config variable `{__key}` for config field `{path_joined}`"
            ) from ex

        raise InvalidConfig(
            f"Missing config variable `{__key}` for `{path_joined}`, this is the {field_desc}"
        )


def load(fp_config):
    with open(fp_config, "r") as fh:
        config_dict = yaml.load(fh, Loader=yaml.FullLoader)
        return Config(config_dict)


def get_config_field(config, field_name, prefix=""):
    if prefix == "":
        value = config.get(field_name)
    else:
        raise NotImplementedError

    if value is None:
        field_desc = FIELD_DESCRIPTIONS[field_name]
        if hasattr(field_desc, "_doc_"):
            field_desc = field_desc._doc_
        raise InvalidConfig(
            f"Field {field_name} is missing from the config. This field describes {field_desc}"
        )
