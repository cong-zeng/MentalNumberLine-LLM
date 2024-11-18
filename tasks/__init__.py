from .comparison import ComparisonTask
from .classifyNum import ClassifyNumTask
from .leftDigit import LeftDigit

TASKS = {
    "comparison": ComparisonTask,
    "classifyNum": ClassifyNumTask,
    "left_digit": LeftDigit,
}