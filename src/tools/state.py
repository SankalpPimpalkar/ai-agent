
class TaskState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.memory_listed = False
        self.tools_listed = False
        self.creates = 0

STATE = TaskState()