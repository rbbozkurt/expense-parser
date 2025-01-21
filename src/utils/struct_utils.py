from enum import Enum

class StrEnum(Enum):
    """
    Custom Enum class to allow direct comparison with strings.

    Methods:
        __eq__: Allows direct comparison of enum values with strings.
        __hash__: Provides hash compatibility for use in dictionaries and sets.
    """
    def __eq__(self, other):
        return self.value == other if isinstance(other, str) else super().__eq__(other)

    def __hash__(self):
        return hash(self.value)
