# Third-party components

This repository's own code is licensed under AGPL-3.0-or-later (`LICENSE`).
Data are vendored only where the license permits and the files are small; larger or
unlicensed datasets are cloned or downloaded by the user.

## Data

| Data | License | Acquisition |
|---|---|---|
| BROAD (consumer MEMS IMU) | CC BY 4.0 | clone `dlaidig/broad` and pull `data_hdf5` (see `X5_broad_real_imu.py`) |
| MUKCa (KUKA/Panda/FR3-unit/Kinova) | MIT | clone `platonics-delft/kinematics_calibration` |
| FR3 / PhilNad (114 poses) | MIT | included at `data/calib_data.pickle`; source `PhilNad/robot-arm-kinematic-calibration`, copyright 2024 Philippe Nadeau; license at `data/FR3_LICENSE`; also downloaded by `X10_fk_calib_real.py` |
| IPA UR10 Series A (laser tracker) | MIT | included at `data/ipa/Series_A`; source `christianlandgraf/ipa-robot-accuracy` |
| pauljac UR5 (1020 poses: 1000 grid + 20 random) | MIT | included at `data/pauljac/UR5`; source `pauljac7700/serial_robots_datasets` |
| GTPN UR5 (727 poses) | CC BY 4.0 | Zenodo 18954650 (`10.5281/zenodo.18954650`); not vendored |
| Baxter dynamics (Zenodo 17035193) | CC BY 4.0 | not vendored |
| ABB RobotCali (draw-wire) | no license | source `Lizhibing1490183152/RobotCali`; **not redistributed**; manual download. Expected 73,894 bytes, SHA256 `724dbbeb90f6469dc1ce9273618df3326a0791aa8a118c79c3c8e59dcf83d7aa`; check with `tools/check_abb_data.py` |

## Software (installed, not vendored)

PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), SciPy (BSD-3-Clause),
h5py (BSD-3-Clause), pandas (BSD-3-Clause), Matplotlib (PSF-based license),
SciencePlots (MIT), roboticstoolbox-python (MIT), robotkinecal (MIT),
vqf (MIT), openpyxl (MIT), rtb-data (pinned 1.1.0; MIT).

`repro/json/` contains our experiment dumps only.
