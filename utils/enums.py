from enum import Enum

class XaiMode(Enum):
  INTERPRETABLE_MODEL = 1
  POST_HOC = 2

class TaskType(Enum):
  CLASSIFICATION = 1
  REGRESSION = 2

class ExplanationScope(Enum):
  GLOBAL = 1
  LOCAL_ROW = 2
  LOCAL_FEATURE = 3

class ResultTypes(Enum):
  FI = 1
  VIS = 2
  TEXT = 3
  EXAMPLE = 3

class Accuracy(Enum):
  LOW = 1
  MEDIUM = 2
  HIGH = 3


class Complexity(Enum):
  LOW = 1
  MEDIUM = 2
  HIGH = 3

class Interactivity(Enum):
  MANUALLY = 1
  YES = 2
  NO = 3

class Interactivity(Enum):
  MANUALLY = 1
  YES = 2
  NO = 3