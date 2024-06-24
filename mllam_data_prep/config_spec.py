from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
import dataclass_wizard

@dataclass
class Range:
    """Defines a range for a variable.

    Attributes:
        start: The start of the range.
        end: The end of the range.
        step: The step size for the range.
    """
    start: str
    end: str
    step: str

@dataclass
class Altitude:
    """Defines altitude information for a variable.

    Attributes:
        sel: The selected altitude levels.
        units: The units of the altitude.
    """
    sel: List[int]
    units: str

@dataclass
class Variable:
    """Defines a variable.

    Attributes:
        altitude: The altitude information for the variable.
    """
    altitude: Altitude = field(default=None)

@dataclass
class DimMapping:
    """Defines a mapping for dimensions.

    Attributes:
        method: The method used for mapping.
        dims: The dimensions to be mapped.
        name_format: The format for naming the mapped dimensions.
    """
    method: str
    dims: Optional[List[str]] = None
    dim: Optional[str] = None
    name_format: str = field(default=None)

@dataclass
class InputDataset:
    """Single input dataset.

    Attributes:
        path: Path to the dataset.
        dims: Expected dimensions of the dataset.
        variables: Variables to select from the dataset.
        dim_mapping: Mapping of the dimensions in the dataset to the dimensions of the architecture's input variables.
        target_architecture_variable: The target architecture variable.
    """
    path: str
    dims: List[str]
    variables: Dict[str, Union[str,Variable]]
    # dim_mapping: Dict[str, Union[str, DimMapping]]
    dim_mapping: Dict[str, DimMapping]
    target_architecture_variable: str

@dataclass
class Architecture:
    """Information about the model architecture this dataset is intended for.

    Attributes:
        input_variables: The input variables.
        input_range: The input range.
        chunking: The chunking information.
    """
    input_variables: Dict[str, List[str]]
    input_range: Dict[str, Range]
    chunking: Dict[str, int]

@dataclass
class Config(dataclass_wizard.YAMLWizard):
    """Configuration for the model.

    Attributes:
        schema_version: Version of the config file schema.
        dataset_version: Version of the dataset itself.
        architecture: Information about the model architecture this dataset is intended for.
        inputs: Input datasets for the model.
    """
    schema_version: str
    dataset_version: str
    architecture: Architecture
    inputs: Dict[str, InputDataset]
    

if __name__ == "__main__":
    config = Config.from_yaml_file("example.danra.yaml")
    import rich
    rich.print(config)