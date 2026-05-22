"""
Unit tests for the ``data`` module using the built-in ``unittest`` framework.

These tests validate:
- CSV bootstrapping into memory (assumed sorted; no expensive merge/sort).
- HL7 parsing for ADT/ORU messages and creatinine extraction.
- Gender extraction from PID-8 and propagation to the inference input.
- Persistence behaviour when the caller (main) explicitly invokes
  ``save_patient_history`` (Alert First, Store Later).
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest


def _import_data_module() -> object:
    """Import and reload ``src.data`` with a clean module state."""
    test_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src import data  # type: ignore
    importlib.reload(data)
    return data


class DataModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        # Backup env vars and isolate storage DB per test
        self._orig_history = os.environ.get("HISTORY_PATH")
        self._orig_dbm = os.environ.get("DBM_PATH")
        os.environ.pop("HISTORY_PATH", None)

        self._tmp = tempfile.TemporaryDirectory()
        os.environ["DBM_PATH"] = os.path.join(self._tmp.name, "patient_history.db")

    def tearDown(self) -> None:
        if self._orig_history is None:
            os.environ.pop("HISTORY_PATH", None)
        else:
            os.environ["HISTORY_PATH"] = self._orig_history

        if self._orig_dbm is None:
            os.environ.pop("DBM_PATH", None)
        else:
            os.environ["DBM_PATH"] = self._orig_dbm

        self._tmp.cleanup()

    def _write_history_file(self, csv_text: str) -> str:
        path = os.path.join(self._tmp.name, "history.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(csv_text)
        return path

    @staticmethod
    def _mk_admit(mrn: str, gender: str = "U", msh_time: str = "202401010000") -> str:
        # PID-8 is gender in our simplified parser.
        return f"MSH|^~\\&|SIM|HOSP|||{msh_time}||ADT^A01|||2.5\rPID|1||{mrn}|||||{gender}"

    @staticmethod
    def _mk_discharge(mrn: str, msh_time: str = "202401020000") -> str:
        return f"MSH|^~\\&|SIM|HOSP|||{msh_time}||ADT^A03|||2.5\rPID|1||{mrn}"

    @staticmethod
    def _mk_oru_creatinine(mrn: str, obr_time: str, value: str, msh_time: str = "202401030000") -> str:
        return (
            f"MSH|^~\\&|SIM|HOSP|||{msh_time}||ORU^R01|||2.5"
            f"\rPID|1||{mrn}"
            f"\rOBR|1||||||{obr_time}"
            f"\rOBX|1|SN|CREATININE||{value}"
        )

    @staticmethod
    def _mk_oru_other(mrn: str, obr_time: str, value: str, msh_time: str = "202401030000") -> str:
        return (
            f"MSH|^~\\&|SIM|HOSP|||{msh_time}||ORU^R01|||2.5"
            f"\rPID|1||{mrn}"
            f"\rOBR|1||||||{obr_time}"
            f"\rOBX|1|SN|SODIUM||{value}"
        )

    def test_initialize_without_history_file(self) -> None:
        data = _import_data_module()
        os.environ["HISTORY_PATH"] = os.path.join(self._tmp.name, "missing.csv")
        data.initialize()
        # ADT should not request inference
        self.assertIsNone(data.process_HL7(self._mk_admit("111", gender="F")))

    def test_initialize_and_creatinine_flow_with_gender(self) -> None:
        data = _import_data_module()
        history_path = self._write_history_file(
            "mrn,timestamp,creatinine\n"
            "123,202401010000,80\n"
            "123,202401010100,85\n"
        )
        os.environ["HISTORY_PATH"] = history_path
        data.initialize()

        # ORU before admit is still parsed and returned by data (no in-hospital gating),
        # but gender will be 'U' until we see an ADT with PID-8.
        res = data.process_HL7(self._mk_oru_creatinine("123", "202401010200", "90"))
        self.assertIsNotNone(res)
        mrn, hist, gender = res  # type: ignore[misc]
        self.assertEqual(mrn, "123")
        self.assertEqual(gender, "U")
        self.assertEqual(hist[-1], ("202401010200", 90.0))

        # Admit with gender
        self.assertIsNone(data.process_HL7(self._mk_admit("123", gender="F")))

        # Next ORU should carry gender 'F'
        res2 = data.process_HL7(self._mk_oru_creatinine("123", "202401010300", "95"))
        self.assertIsNotNone(res2)
        mrn2, hist2, gender2 = res2  # type: ignore[misc]
        self.assertEqual(mrn2, "123")
        self.assertEqual(gender2, "F")
        self.assertEqual(hist2[-1], ("202401010300", 95.0))

    def test_non_creatinine_result_ignored(self) -> None:
        data = _import_data_module()
        data.initialize()
        self.assertIsNone(data.process_HL7(self._mk_admit("999", gender="M")))
        self.assertIsNone(data.process_HL7(self._mk_oru_other("999", "202401010200", "135")))

    def test_discharge_does_not_break_history(self) -> None:
        """Data no longer gates inference by admission; discharge should not crash and history should still append."""
        data = _import_data_module()
        data.initialize()
        data.process_HL7(self._mk_admit("789", gender="M"))

        res1 = data.process_HL7(self._mk_oru_creatinine("789", "202401010200", "100"))
        self.assertIsNotNone(res1)

        # Discharge
        self.assertIsNone(data.process_HL7(self._mk_discharge("789")))

        # Another ORU still appends and returns history
        res2 = data.process_HL7(self._mk_oru_creatinine("789", "202401010300", "110"))
        self.assertIsNotNone(res2)
        mrn, hist, gender = res2  # type: ignore[misc]
        self.assertEqual(mrn, "789")
        self.assertEqual(gender, "M")
        self.assertEqual(hist[-1], ("202401010300", 110.0))

    def test_persistence_across_initialize_requires_explicit_save(self) -> None:
        """Persistence happens when the caller invokes save_patient_history (Alert First, Store Later)."""
        data = _import_data_module()
        os.environ["HISTORY_PATH"] = os.path.join(self._tmp.name, "missing.csv")
        data.initialize()

        data.process_HL7(self._mk_admit("555", gender="F"))
        res = data.process_HL7(self._mk_oru_creatinine("555", "202401020000", "95"))
        self.assertIsNotNone(res)
        mrn, hist, gender = res  # type: ignore[misc]
        self.assertEqual(mrn, "555")
        self.assertEqual(gender, "F")
        self.assertEqual(hist, [("202401020000", 95.0)])

        # Explicitly persist (simulating main saving after alert)
        data.save_patient_history("555")

        # Reload module to clear memory and re-initialize from DBM
        data = _import_data_module()
        os.environ["HISTORY_PATH"] = os.path.join(self._tmp.name, "missing.csv")
        data.initialize()

        # Send another ORU; history should include the previously persisted value
        res2 = data.process_HL7(self._mk_oru_creatinine("555", "202401030000", "100"))
        self.assertIsNotNone(res2)
        mrn2, hist2, gender2 = res2  # type: ignore[misc]
        self.assertEqual(mrn2, "555")
        # gender may be 'U' unless we admit again with PID-8; re-admit to restore gender.
        # We admit to make gender stable for the pipeline.
        data.process_HL7(self._mk_admit("555", gender="F"))
        res3 = data.process_HL7(self._mk_oru_creatinine("555", "202401040000", "105"))
        self.assertIsNotNone(res3)
        mrn3, hist3, gender3 = res3  # type: ignore[misc]
        self.assertEqual(mrn3, "555")
        self.assertEqual(gender3, "F")
        self.assertIn(("202401020000", 95.0), hist3)
        self.assertEqual(hist3[-1], ("202401040000", 105.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
