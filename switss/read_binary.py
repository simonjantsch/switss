from struct import *
import socket
import pathlib


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
        for _ in range(action_count):
            (distribution_size,) = self._recv(Exchange.distribution_message)
            if distribution_size == 0:
                if state in self.system:
                    del self.system[state]
            else:
                self.system[state] = {
                    t: p
                    for t, p in Exchange.transition_message.iter_unpack(
                        self._recv_raw(
                            Exchange.transition_message.size * distribution_size
                        )
                    )
                }

    def loop(self):
        (message_type,) = self._recv(Exchange.header)

        if message_type == 0:
            # Init
            self.system = dict()
            self._read_state()
        elif message_type == 1:
            # Update
            (modified_count,) = self._recv(Exchange.update_message)
            for _ in range(modified_count):
                self._read_state()
        elif message_type == 2:
            # Compute bounds
            upper_bounds = {s: 1.0 for s in self.system.keys()}
            found_core = set()

            self._send(Exchange.compute_response.pack(len(upper_bounds), len(found_core)))
            for s, b in upper_bounds.items():
                self._send(Exchange.state_bounds.pack(s, b))
            for s in found_core:
                self._send(Exchange.core_state.pack(s))
            self.c.flush()


if __name__ == "__main__":
    ex = Exchange.make()
    while True:
        ex.loop()
