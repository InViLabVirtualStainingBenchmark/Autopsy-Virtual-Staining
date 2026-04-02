# REFACTOR: watcher.py was not committed to the original repo but is imported by
# train_stage2_seperate_train_by_iters.py. Watcher.check_stop() is called throughout
# the training loop to allow graceful keyboard-interrupt handling.
# This stub makes the import succeed. Ctrl+C will still stop training via Python's
# default SIGINT handling.
class Watcher:
    def check_stop(self):
        pass
