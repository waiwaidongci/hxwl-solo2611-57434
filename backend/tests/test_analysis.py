import unittest
from datetime import date, timedelta

from src.services import analysis


PUBLISH = "2026-11-21T08:00:00+08:00"


def arrival(batch, day, features, received=None):
    return {
        "sort_date": day,
        "batch_id": batch,
        "received_at": received or f"{day}T07:00:00+08:00",
        "mixed_appearance_pct": features[0],
        "unreadable_label_pct": features[1],
        "belt_load_pct": features[2],
        "moisture_pct": features[3],
        "inspection_version": "",
        "inspection_closed_at": "",
        "mis_sorted": "",
    }


def inspection(batch, day, result, version=1, closed=None):
    return {
        "sort_date": day,
        "batch_id": batch,
        "received_at": "",
        "mixed_appearance_pct": "",
        "unreadable_label_pct": "",
        "belt_load_pct": "",
        "moisture_pct": "",
        "inspection_version": str(version),
        "inspection_closed_at": closed or f"{day}T20:00:00+08:00",
        "mis_sorted": str(result),
    }


def history():
    rows = []
    start = date(2026, 11, 9)
    for offset in range(12):
        day = (start + timedelta(days=offset)).isoformat()
        for index in range(1, 5):
            batch = f"HS-{day[5:7]}{day[8:10]}-{index:02d}"
            heavy = index % 2 == 0
            features = ("40" if heavy else "10", "20", "60", "15")
            rows.append(arrival(batch, day, features))
            rows.append(inspection(batch, day, 1 if heavy else 0))
    return rows


def shift():
    rows = []
    for index, feature in enumerate(
        (("45", "25", "60", "20"), ("35", "20", "65", "18"), ("25", "15", "70", "15"),
         ("15", "10", "80", "10")), start=1
    ):
        rows.append(arrival(f"HS-1121-{index:02d}", "2026-11-21", feature))
    return rows


def params(**overrides):
    base = {"publish_at": "2026-11-21T08:00", "review_quota": "2"}
    base.update(overrides)
    return base


class RosterTest(unittest.TestCase):
    def test_publish_freezes_participants_and_selection(self):
        rows = history() + shift()
        result = analysis.analyze(rows, params())
        roster = result["roster"]
        self.assertEqual([entry["batch_id"] for entry in roster["entries"]],
                         ["HS-1121-01", "HS-1121-02", "HS-1121-03", "HS-1121-04"])
        self.assertEqual([entry["selected"] for entry in roster["entries"]],
                         [True, True, False, False])

        late = rows + [arrival("HS-1121-99", "2026-11-21",
                               ("99", "99", "99", "99"), "2026-11-21T09:00:00+08:00")]
        replayed = analysis.replay(roster, late)
        self.assertEqual(replayed["roster"]["entries"], roster["entries"])
        self.assertEqual([entry["selected"] for entry in replayed["roster"]["entries"]],
                         [True, True, False, False])
        self.assertIn("HS-1121-99", replayed["roster"]["late_arrivals"])

    def test_ties_break_by_batch_id(self):
        rows = history()
        for batch in ("HS-1121-02", "HS-1121-01"):
            rows.append(arrival(batch, "2026-11-21", ("45", "25", "60", "20")))
        roster = analysis.analyze(rows, params(review_quota="1"))["roster"]
        self.assertEqual([entry["batch_id"] for entry in roster["entries"]],
                         ["HS-1121-01", "HS-1121-02"])
        self.assertEqual([entry["probability"] for entry in roster["entries"]],
                         [roster["entries"][0]["probability"]] * 2)
        self.assertEqual([entry["selected"] for entry in roster["entries"]], [True, False])

    def test_newest_version_wins_when_reported_out_of_order(self):
        rows = history() + shift()
        rows.append(inspection("HS-1121-01", "2026-11-21", 1, version=2,
                               closed="2026-11-21T21:30:00+08:00"))
        rows.append(inspection("HS-1121-01", "2026-11-21", 0, version=1))
        current = analysis.analyze(rows, params())["tables"][3]["rows"]
        self.assertEqual(current[0], ["抽检发现错投", 1, 0])

    def test_same_version_conflict_is_pending(self):
        rows = history() + shift()
        rows.append(inspection("HS-1121-01", "2026-11-21", 0, version=2,
                               closed="2026-11-21T20:30:00+08:00"))
        rows.append(inspection("HS-1121-01", "2026-11-21", 1, version=2,
                               closed="2026-11-21T21:00:00+08:00"))
        result = analysis.analyze(rows, params())
        summary = {row["label"]: row["value"] for row in result["metrics"]}
        self.assertEqual(summary["待核"], 1)
        self.assertEqual(summary["当前命中"], 0)
        self.assertEqual(summary["当前误报"], 0)
        pending = result["tables"][2]["rows"]
        self.assertEqual(pending[0][1], "HS-1121-01")

    def test_uninspected_batches_are_listed_separately(self):
        rows = history() + shift()
        rows.append(inspection("HS-1121-02", "2026-11-21", 1))
        result = analysis.analyze(rows, params())
        summary = {row["label"]: row["value"] for row in result["metrics"]}
        self.assertEqual(summary["未抽检"], 3)
        self.assertEqual(summary["当前命中"], 1)
        self.assertEqual(summary["当前误报"], 0)
        self.assertEqual(len(result["tables"][1]["rows"]), 3)

    def test_duplicate_rows_count_once(self):
        rows = history() + shift()
        once = analysis.analyze(rows, params())
        twice = analysis.analyze(rows + rows, params())
        self.assertEqual(once["metrics"], twice["metrics"])
        self.assertEqual(len(once["roster"]["entries"]), len(twice["roster"]["entries"]))

    def test_history_correction_updates_replay_not_selection(self):
        rows = history() + shift()
        original = analysis.analyze(rows, params())
        roster = original["roster"]
        corrected = rows + [inspection("HS-1121-01", "2026-11-21", 0, version=2,
                                       closed="2026-11-22T09:00:00+08:00")]
        replayed = analysis.replay(roster, corrected)
        self.assertEqual(replayed["roster"]["entries"], roster["entries"])
        summary = {row["label"]: row["value"] for row in replayed["metrics"]}
        self.assertEqual(summary["当前命中"], 0)
        self.assertEqual(summary["当前误报"], 1)

    def test_training_ignores_inspections_closed_after_publish(self):
        rows = history() + shift()
        rows.append(inspection("HS-1121-01", "2026-11-21", 1, version=2,
                               closed="2026-11-21T09:00:00+08:00"))
        rows.append(inspection("HS-1110-01", "2026-11-10", 1, version=2,
                               closed="2026-11-21T10:00:00+08:00"))
        before = analysis.analyze(rows, params())
        self.assertEqual(before["roster"]["training"]["batches"], 48)


if __name__ == "__main__":
    unittest.main()
