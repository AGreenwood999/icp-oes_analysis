from pathlib import Path
import numpy as np
import polars as pl
from icp_oes_analysis.core import Experiment


class Plate:
    def __init__(self, file: Path) -> None:
        self.raw_data = pl.scan_csv(file)
        self.calib_conc = np.repeat(np.array([0, 0.05, 0.1, 0.2, 0.5, 1, 2, 4]), 3)
        self.calibration_data = (
            self.raw_data.select("10", "11", "12").collect().to_numpy().flatten()
        )
        self.std_04_signal = self.raw_data.select("9").collect().to_numpy().flatten()
        self.std_02_signal = self.raw_data.select("8").collect().to_numpy().flatten()
        self.std_015_signal = self.raw_data.select("7").collect().to_numpy().flatten()
        self.std_01_signal = self.raw_data.select("6").collect().to_numpy().flatten()
        self.std_005_signal = self.raw_data.select("5").collect().to_numpy().flatten()
        self.blanks_signal = np.concatenate(
            [
                self.raw_data.select("4").collect().to_numpy().flatten(),
                self.raw_data.filter(pl.col("").is_in(["A", "B", "C"]))
                .select("3")
                .collect()
                .to_numpy()
                .flatten(),
            ]
        )

        self.slope, self.intercept, self.r2 = self._fit_standards()

        self.blanks_conc = (self.blanks_signal - self.intercept) / self.slope
        self.std_04_conc = (self.std_04_signal - self.intercept) / self.slope
        self.std_02_conc = (self.std_02_signal - self.intercept) / self.slope
        self.std_015_conc = (self.std_015_signal - self.intercept) / self.slope
        self.std_01_conc = (self.std_01_signal - self.intercept) / self.slope
        self.std_005_conc = (self.std_005_signal - self.intercept) / self.slope

    def calculate_concentration(self, data: np.ndarray):
        out = np.empty_like(data)
        for i, d in enumerate(data):
            out[i] = (d - self.intercept) / self.slope

        return out

    def _fit_standards(self) -> tuple[np.float64, np.float64, np.float64]:
        m, b = np.polyfit(self.calib_conc, self.calibration_data, 1)
        f = m * self.calib_conc + b

        ssres = np.sum([(fi - yi) ** 2 for (fi, yi) in zip(f, self.calibration_data)])
        sstot = np.sum(
            [(yi - np.mean(self.calibration_data)) ** 2 for yi in self.calibration_data]
        )
        r2 = 1 - ssres / sstot

        return m, b, r2

    raw_data: pl.LazyFrame
    calibration_data: np.ndarray
    calib_conc: np.ndarray
    std_04_signal: np.ndarray
    std_04_conc: np.ndarray
    std_02_signal: np.ndarray
    std_02_conc: np.ndarray
    std_015_signal: np.ndarray
    std_015_conc: np.ndarray
    std_01_signal: np.ndarray
    std_01_conc: np.ndarray
    std_005_signal: np.ndarray
    std_005_conc: np.ndarray
    blanks_signal: np.ndarray
    blanks_conc: np.ndarray
    slope: float
    intercept: float
    r2: float
    lob: np.floating


DATA_FILES: list[Path] = [
    Path(
        "/home/augustus/UNandUP/theralode-code/EXP_202511_FerrozineLimitOfQuantification/data/plate1_ferrozinelimitofquantification_20251106.csv"
    ),
    Path(
        "/home/augustus/UNandUP/theralode-code/EXP_202511_FerrozineLimitOfQuantification/data/plate2_ferrozinelimitofquantification_20251106.csv"
    ),
    Path(
        "/home/augustus/UNandUP/theralode-code/EXP_202511_FerrozineLimitOfQuantification/data/plate3_ferrozinelimitofquantification_20251106.csv"
    ),
    Path(
        "/home/augustus/UNandUP/theralode-code/EXP_202511_FerrozineLimitOfQuantification/data/plate4_ferrozinelimitofquantification_20251106.csv"
    ),
    Path(
        "/home/augustus/UNandUP/theralode-code/EXP_202511_FerrozineLimitOfQuantification/data/plate5_ferrozinelimitofquantification_20251106.csv"
    ),
]


def extension(exp: Experiment):
    data: list[Plate] = [Plate(file) for file in DATA_FILES]

    blanks = np.concatenate([d.blanks_conc for d in data])
    lob = np.mean(blanks) + 1.645 * np.std(blanks, ddof=1)
    print("LOB")
    print("mean", np.mean(blanks))
    print("std", np.std(blanks))
    print(lob)

    detection_samples = np.array(
        [
            np.concatenate([d.std_02_conc for d in data]),
            np.concatenate([d.std_015_conc for d in data]),
            np.concatenate([d.std_01_conc for d in data]),
            np.concatenate([d.std_005_conc for d in data]),
        ]
    )

    detection_SDS = np.sqrt(
        (1 / 4) * np.sum([np.var(s, ddof=1) for s in detection_samples])
    )

    lod = lob + 1.645 / (1 - (1 / (4 * 156))) * detection_SDS
    print("LOD")
    print("detection SDS", detection_SDS)
    print(lod)

    icp_oes_conc = exp.results.select("Fe_concentration_mgL").to_numpy().flatten()
    ferrozine_conc = np.empty((5, 40), dtype=np.float64)
    ferrozine_conc[0, :] = np.concatenate([d.std_04_conc for d in data])
    ferrozine_conc[1, :] = np.concatenate([d.std_02_conc for d in data])
    ferrozine_conc[2, :] = np.concatenate([d.std_015_conc for d in data])
    ferrozine_conc[3, :] = np.concatenate([d.std_01_conc for d in data])
    ferrozine_conc[4, :] = np.concatenate([d.std_005_conc for d in data])

    loq = [
        (np.abs(np.mean(ferr) - icp) + 2 * np.std(ferr)) / np.mean(ferr)
        for (ferr, icp) in zip(ferrozine_conc, icp_oes_conc)
    ]

    print(
        [
            (np.abs(np.mean(ferr) - icp)) / np.mean(ferr)
            for (ferr, icp) in zip(ferrozine_conc, icp_oes_conc)
        ]
    )

    print(
        [
            (2 * np.std(ferr)) / np.mean(ferr)
            for (ferr, icp) in zip(ferrozine_conc, icp_oes_conc)
        ]
    )

    print(loq)
