import tempfile

import mllam_data_prep as mdp
import tests.data as testdata
from mllam_data_prep.ops import cropping


def test_create_convex_hull_mask():
    tmpdir = tempfile.TemporaryDirectory()
    domain_size = 500 * 1.0e3  # length and width of domain in meters
    config_lam = testdata.create_input_datasets_and_config(
        identifier="lam",
        data_categories=["state"],
        tmpdir=tmpdir,
        xlim=[-domain_size / 2.0, domain_size / 2.0],
        ylim=[-domain_size / 2.0, domain_size / 2.0],
    )
    # make the global domain twice as large as the LAM domain so that the lam
    # domain is contained within the global domain
    config_global = testdata.create_input_datasets_and_config(
        identifier="global",
        data_categories=["state"],
        tmpdir=tmpdir,
        xlim=[-domain_size, domain_size],
        ylim=[-domain_size, domain_size],
    )

    ds_lam = mdp.create_dataset(config=config_lam)
    ds_global = mdp.create_dataset(config=config_global)

    cropping.create_convex_hull_mask(ds=ds_global, ds_reference=ds_lam)
