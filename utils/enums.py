from enum import Enum

class XaiMode(Enum):
  INTERPRETABLE_MODEL = 1
  POST_HOC = 2

class TaskType(Enum):
  CLASSIFICATION = 1
  REGRESSION = 2

class ExplanationScope(Enum):
  GLOBAL = 1
  LOCAL = 2