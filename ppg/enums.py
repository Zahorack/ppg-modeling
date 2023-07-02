from enum import IntEnum, unique


@unique
class WavelengthIndex(IntEnum):
    PLETH = 0
    RED = 1
    IR = 2


class DataColumn:
    WAVELENGTH = 'wavelength'
    TIME = 'time'
    PPG_CURVE = 'PPG curve'
