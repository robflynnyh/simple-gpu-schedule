import json
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock


WITH_GPU = SourceFileLoader(
    "with_gpu",
    str(Path(__file__).resolve().parents[1] / "with-gpu"),
).load_module()


def ticket(ticket_id, pool_indices, num=1):
    return {
        "id": ticket_id,
        "pool_indices": pool_indices,
        "num": num,
    }


class FifoPlanTests(unittest.TestCase):
    def test_blocked_specific_gpu_does_not_block_disjoint_gpu(self):
        tickets = [
            ticket("gpu1", [1]),
            ticket("gpu2", [2]),
        ]

        self.assertEqual(WITH_GPU.fifo_plan(tickets, [2]), {"gpu2": [2]})

    def test_blocked_any_ticket_blocks_overlapping_specific_gpu(self):
        tickets = [
            ticket("any", [0, 1, 2], num=2),
            ticket("gpu2", [2]),
        ]

        self.assertEqual(WITH_GPU.fifo_plan(tickets, [2]), {})

    def test_any_ticket_runs_before_later_specific_gpu(self):
        tickets = [
            ticket("any", [0, 1, 2]),
            ticket("gpu2", [2]),
        ]

        self.assertEqual(WITH_GPU.fifo_plan(tickets, [2]), {"any": [2]})

    def test_blocked_multi_gpu_ticket_allows_disjoint_later_ticket(self):
        tickets = [
            ticket("gpu1_and_gpu2", [1, 2], num=2),
            ticket("gpu3", [3]),
        ]

        self.assertEqual(WITH_GPU.fifo_plan(tickets, [2, 3]), {"gpu3": [3]})

    def test_later_ticket_can_use_unclaimed_gpu_from_overlapping_pool(self):
        tickets = [
            ticket("gpu1_and_gpu2", [1, 2], num=2),
            ticket("gpu2_or_gpu3", [2, 3]),
        ]

        self.assertEqual(WITH_GPU.fifo_plan(tickets, [2, 3]), {"gpu2_or_gpu3": [3]})


class CliTests(unittest.TestCase):
    def test_default_idle_seconds_is_30(self):
        args = WITH_GPU.parse_cli(["any", "--", "true"])

        self.assertEqual(args.idle_seconds, 30)
        self.assertFalse(args.status)

    def test_status_does_not_require_command(self):
        args = WITH_GPU.parse_cli(["--status"])

        self.assertTrue(args.status)
        self.assertEqual(args.pool, "any")
        self.assertEqual(args.cmd, [])

    def test_status_accepts_pool(self):
        args = WITH_GPU.parse_cli(["1,2", "--status"])

        self.assertTrue(args.status)
        self.assertEqual(args.pool, "1,2")

    def test_all_pool_alias_matches_any(self):
        self.assertEqual(WITH_GPU.parse_pool("all", [0, 1, 2]), [0, 1, 2])


class StaleStateTests(unittest.TestCase):
    def make_record(self, directory, **overrides):
        path = Path(directory) / "record.json"
        record = {
            "host": WITH_GPU.socket.gethostname(),
            "boot_id": "current-boot",
            "pid": 123,
            "pid_start_ticks": 456,
            "_path": str(path),
        }
        record.update(overrides)
        path.write_text(json.dumps(record))
        return record, path

    def test_dead_running_record_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            record, path = self.make_record(directory)
            with mock.patch.object(WITH_GPU, "pid_alive", return_value=False):
                live = WITH_GPU.remove_stale_running(
                    [record],
                    "current-boot",
                )

            self.assertEqual(live, [])
            self.assertFalse(path.exists())

    def test_reused_pid_running_record_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            record, path = self.make_record(directory)
            with (
                mock.patch.object(WITH_GPU, "pid_alive", return_value=True),
                mock.patch.object(
                    WITH_GPU,
                    "process_start_ticks",
                    return_value=999,
                ),
            ):
                live = WITH_GPU.remove_stale_running(
                    [record],
                    "current-boot",
                )

            self.assertEqual(live, [])
            self.assertFalse(path.exists())

    def test_matching_running_process_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            record, path = self.make_record(directory)
            with (
                mock.patch.object(WITH_GPU, "pid_alive", return_value=True),
                mock.patch.object(
                    WITH_GPU,
                    "process_start_ticks",
                    return_value=456,
                ),
            ):
                live = WITH_GPU.remove_stale_running(
                    [record],
                    "current-boot",
                )

            self.assertEqual(live, [record])
            self.assertTrue(path.exists())

    def test_previous_boot_record_is_removed_without_pid_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            record, path = self.make_record(
                directory,
                boot_id="old-boot",
            )
            with mock.patch.object(WITH_GPU, "pid_alive") as pid_alive:
                live = WITH_GPU.remove_stale_running(
                    [record],
                    "current-boot",
                )

            self.assertEqual(live, [])
            self.assertFalse(path.exists())
            pid_alive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
