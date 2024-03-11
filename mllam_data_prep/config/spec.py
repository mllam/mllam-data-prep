FIELD_DESCRIPTIONS = dict(
    schema_version="version of the config file schema",
    dataset_version="version of the dataset itself",
    architecture=dict(
        _doc_="Information about the model architecture this dataset is intended for",
        sampling_dim="dimension to sample along",
        input_variables=dict(
            _doc_="Variables the architecture expects as input",
            _values_="Dimensions for each input variable (e.g. [time, x, y])",
        ),
        input_range=dict(
            _doc_="Value range for the coordinates that span the input variables (e.g. time)",
            _values_=dict(
                _doc_="Value range for the coordinate",
                start="start of the range",
                end="end of the range",
                step="step size of the range",
            ),
        ),
        chunking=dict(
            _doc_="Chunking for dimensions of the input variables",
            _values_=dict(
                _doc_="Chunking for a single input variable dimension",
            ),
        ),
    ),
    inputs=dict(
        _doc_="Input datasets for the model",
        _values_=dict(
            _doc_="single input dataset",
            path="path to the dataset",
            dims="dimensions of the dataset",
            attributes="attributes the dataset should have",
            variables=dict(
                _doc_="variables to select from the dataset",
                _values_=dict(
                    _doc_="coordinate selections for the variable",
                    _values_=dict(
                        _doc_="selection along a single coordinate",
                        sel="coordinate values to select from the variable",
                        units="units of the variable",
                    ),
                ),
            ),
            dim_mapping=dict(
                _doc_="mapping of the dimensions in the dataset to the dimensions of the architecture's input variables",
                _values_=dict(
                    _doc_="mapping of the dimensions in the input dataset to the architecture's input variables",
                    method="method by which mapping is done (e.g. 'flatten', 'stack_variables_by_var_name')",
                    dims="dimensions in the source dataset to map from",
                    name_format="string-format for mapped variable (used when stacking variables to coordinate values)",
                ),
            ),
            target="name of variable in the output that the given input should stored in",
        ),
    ),
)
