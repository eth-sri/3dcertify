from time import perf_counter

BOUND_TIMER = "bounds"
TOTAL_TIMER = "total"


class Timer:
    __DEFAULT_KEY = "__DEFAULT__"

    def __init__(self):
        self.starts = {}
        self.totals = {}

    def start(self, key: str = __DEFAULT_KEY):
        if key in self.starts:
            raise Exception(f"Timer '{key}' is already running")

        self.starts[key] = perf_counter()

    def stop(self, key: str = __DEFAULT_KEY):
        if key not in self.starts:
            raise Exception(f"Timer '{key}' is not running")

        elapsed_time = perf_counter() - self.starts.pop(key)
        total_time = elapsed_time + self.totals.get(key, 0.0)
        self.totals[key] = total_time
        return elapsed_time

    def get(self, key: str = __DEFAULT_KEY, default: float = None):
        if key not in self.totals and default is None:
            raise Exception(f"Timer '{key}' does not exist")
        return self.totals.get(key, default)
