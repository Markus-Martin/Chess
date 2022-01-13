

class MissingPieceError(Exception):
    """
    Raised when there's no piece to move at the given location.
    """
    def __init__(self, state, location, msg="Missing piece at the location"):
        super().__init__(msg)
        self.state = state
        self.location = location
        self.msg = msg

    def __str__(self):
        return f'{self.msg} {self.location}\n{self.state}'
