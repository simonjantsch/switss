


class ReachabilityForm:
    def __init__(self, P, initial, to_target, index_by_state_action):
        self.P = P
        self.initial = initial
        self.to_target = to_target
        self.index_by_state_action = index_by_state_action

    