import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
