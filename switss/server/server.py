from struct import *
import socket
import pathlib
import numpy as np
import scipy as sp
from bidict import bidict

class Exchange(object):
    header = Struct(">i")
    init_message = Struct(">i")
    update_message = Struct(">i")
    state_message = Struct(">ii")
    distribution_message = Struct(">i")
    transition_message = Struct(">id")

    compute_response = Struct(">ii")
    state_bounds = Struct(">id")
    core_state = Struct(">i")

    @staticmethod
    def make(path=None):
        if path is None:
            path = pathlib.Path("/tmp/core.sock")
        if path.exists():
            path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(path))
        server.listen(1)
        conn, addr = server.accept()
        return Exchange(conn)

    def __init__(self, c):
        self.c = c

    def _recv_raw(self, size):
        return self.c.recv(size, socket.MSG_WAITALL)

    def _recv(self, struct):
        return struct.unpack(self._recv_raw(struct.size))

    def _send(self, data):
        self.c.sendall(data)

    def _read_state(self):
        state, action_count = self._recv(Exchange.state_message)
        if state not in seen_states:
            self.seen_states.insert(state)
            self.ext_to_int_state[state] = self.nr_states
            self.nr_states += 1

        ## why not model delete by action_count=0 ?
        for a in range(action_count):
            (distribution_size,) = self._recv(Exchange.distribution_message)
            if distribution_size == 0:
                self._delete_state(state)
            else:
                new_transitions = {
                    t: p
                    for t, p in Exchange.transition_message.iter_unpack(
                        self._recv_raw(
                            Exchange.transition_message.size * distribution_size
                        )
                    )
                }
                if (action_count == 1) and (new_transitions[state] == 1):
                    self.goal_states.insert(state)
                list_idx = self._get_list_idx(state,a)
                self.transitions[list_idx] = new_transitions

    def _delete_state(self,state):
        a = 0
        while true:
            if (state,a) not in self.list_idx_to_ext_sap.inv.keys():
                break
            self.list_idx_to_ext_sap.inv[(state,a)] = dict()
            a += 1

    def _get_list_idx(self,state, action_id):
        if (state,action_id) in self.list_idx_to_ext_sap.inv.keys():
            return self.list_idx_to_ext_sap.inv[(state,action_id)]
        else:
            transitions.append(dict())
            self.nr_sap += 1
            return len(transitions) - 1

    def _to_matrix(self):
        res = sp.dok_matrix(self.nr_sap, self.nr_states)
        index_by_state_action = bidict()

        for idx in range(len(self.transitions)):
            if len(self.transitions[idx]) == 0:
                continue
            ext_state, action_id = self.list_idx_to_ext_sap[idx]
            from_idx = self.ext_to_int_state[ext_state]
            index_by_state_action[from_idx,action_id] = idx
            for to_ext, p in self.transitions[idx]:
                to_idx = self.ext_to_int_state[to_ext]
                res[idx, to_idx] = p

        return res, index_by_state_action

    def loop(self):
        (message_type,) = self._recv(Exchange.header)

        if message_type == 0:
            # Init
            self.transitions = []
            self.list_idx_to_ext_sap = bidict()
            self.nr_states = 0
            self.nr_sap = 0
            self.ext_to_int_state = bidict()
            self.seen_states = set()
            self.goal_states = set()

            self._read_state()

        elif message_type == 1:
            # Update
            (modified_count,) = self._recv(Exchange.update_message)
            for _ in range(modified_count):
                self._read_state()

        elif message_type == 2:
            # Compute bounds
            upper_bounds = {s: 1.0 for s in self.system.keys()}
            P, index_by_state_action = self._to_matrix()
            mdp = MDP( P, index_by_state_action, dict([("goal", self.goal_states)]))
            rf = ReachabilityForm.reduce( mdp, "init", "goal" )
            
            heur = QSHeur(solver="gurobi")
            result = heur.solve(rf, 0.999, "min")
            if self.nr_sap == self.nr_states:
                result2 = heur.solve(rf, 0.999, "max")
                nr_states_result = np.sum(result.subsystem.subsystem_mask)
                nr_states_result2 = np.sum(result2.subsystem.subsystem_mask)
                if nr_states_result2 < nr_states_result:
                    result = result2
                    nr_states_result = nr_states_result2
                    
            self._send(Exchange.compute_response.pack(len(upper_bounds), nr_states_result))
            for s, b in upper_bounds.items():
                self._send(Exchange.state_bounds.pack(s, b))
            for state_idx in range(self.nr_states):
                if result.subsystem.subsystem_mask[state_idx]:
                    self._send(Exchange.core_state.pack(ext_to_int_state.inv[state_idx]))

            self.c.flush()


def start_server():
    print("started server, waiting for messages ")
    ex = Exchange.make()
    print("info: made exchange object ")
    while True:
        ex.loop()


