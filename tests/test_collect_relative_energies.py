import json

from dopingflow.collect import read_formation_meta


def test_collect_flattens_formation_stage_relative_columns(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text(
        json.dumps(
            {
                "primary_reference_label": "SbO2",
                "reference_results": {
                    "SbO2": {
                        "E_form_eV_per_cation": 0.20,
                        "mixing": {"E_mix_eV_per_cation": 0.30},
                        "relative": {
                            "endpoint_x": 1.0,
                            "reference": "oxide_reference_already_tieline_corrected",
                            "E_form_rel_eV_per_cation": 0.20,
                            "E_mix_rel_eV_per_cation": 0.30,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = read_formation_meta(path)
    wide = result["wide_reference_results"]

    assert result["primary_reference_label"] == "SbO2"
    assert wide["E_form_rel_eV_per_cation__SbO2"] == 0.20
    assert wide["E_mix_rel_eV_per_cation__SbO2"] == 0.30
    assert wide["relative_reference__SbO2"] == "oxide_reference_already_tieline_corrected"
