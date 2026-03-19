# emulators-demo

[![DOI](https://zenodo.org/badge/DOI/19111360.svg)](https://doi.org/19111360)

This repository contains workflow scripts and example cases to demonstrate emulator-assisted modelling workflows using pyEMU and PEST++.

## Contents

The repository includes the following primary workflow scripts:

*   `workflow_gpr.py`: Contains examples utilizing Gaussian Process Regression (GPR).
*   `workflow_dsi.py`: Contains examples utilizing Data-Space Inversion (DSI).

## Installation

To set up the required environment, use the provided `environment.yml` file.

1.  Create the environment:
    ```bash
    conda env create -f environment.yml
    ```

2.  Activate the environment:
    ```bash
    conda activate emul
    ```

## Usage

To run the example workflows, ensure the environment is activated and execute the Python scripts from the root directory:

To run the GPR examples:
```bash
python workflow_gpr.py
```

To run the DSI examples:
```bash
python workflow_dsi.py
```

## Citation

If you use this software, please cite it using the DOI badge above. Metadata for Zenodo is maintained in [.zenodo.json](.zenodo.json).

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
