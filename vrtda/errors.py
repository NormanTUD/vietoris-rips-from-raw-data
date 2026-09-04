class VrtdaError(Exception):
    pass


class ShapeError(VrtdaError):
    pass


class DataError(VrtdaError):
    pass


class MetricError(VrtdaError):
    pass


class FiltrationError(VrtdaError):
    pass


class TooLargeError(VrtdaError):
    pass
