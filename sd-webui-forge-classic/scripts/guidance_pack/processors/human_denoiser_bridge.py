import threading


def _shared_state_interrupted() -> bool:
    """Read Forge's interrupted flag lazily; default False when unavailable."""
    try:
        from modules import shared

        return bool(getattr(getattr(shared, "state", None), "interrupted", False))
    except Exception:
        return False


class HumanDenoiserBridge:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_instance()
        return cls._instance

    def _init_instance(self):
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.choice_idx = None
        self.waiting_info = None
        self.selected_idx = None
        self.walk_remaining = 0
        self.walk_path = 0

    def store_selection(self, idx):
        with self.lock:
            self.selected_idx = int(idx)

    def get_selection(self):
        with self.lock:
            return self.selected_idx

    def set_walk(self, steps, path_idx):
        with self.lock:
            self.walk_remaining = max(0, int(steps))
            self.walk_path = int(path_idx)

    def consume_walk(self):
        with self.lock:
            if self.walk_remaining > 0:
                self.walk_remaining -= 1
                return True, self.walk_path
            return False, 0

    def wait_for_choice(self, step, candidate_paths, total, sigma):
        self.waiting_info = {
            "step": step,
            "total": total,
            "sigma": sigma,
            "paths": candidate_paths,
            "n": len(candidate_paths),
        }
        self.choice_idx = None
        self.event.clear()
        while not self.event.wait(timeout=0.5):
            if _shared_state_interrupted():
                return -1
        with self.lock:
            val = self.choice_idx
            self.choice_idx = None
        return val if val is not None else 0

    def submit_choice(self, step, chosen_index):
        with self.lock:
            self.choice_idx = int(chosen_index)
        self.event.set()

    def signal_stop(self):
        with self.lock:
            self.choice_idx = -1
        self.event.set()

    def get_latest_waiting(self):
        return self.waiting_info

    def cleanup(self):
        self.waiting_info = None
        self.choice_idx = None
        self.event.clear()
        with self.lock:
            self.selected_idx = None
            self.walk_remaining = 0
            self.walk_path = 0
