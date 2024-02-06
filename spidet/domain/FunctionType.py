from enum import Enum


class FunctionType(Enum):
    H_COEFFICIENTS = "H_best"
    STD_LINE_LENGTH = "std_line_length"

    def name(self) -> str:
        return self.value

    @classmethod
    def from_file_path(cls, file_path: str):
        if cls.STD_LINE_LENGTH.name() in file_path:
            return cls.STD_LINE_LENGTH
        elif cls.H_COEFFICIENTS.name() in file_path:
            return cls.H_COEFFICIENTS
