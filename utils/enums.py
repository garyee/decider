from enum import Enum

class XaiMode(Enum):
  INTERPRETABLE_MODEL = 1
  POST_HOC = 2

  def getName(self=None):
    nameArray= {
      XaiMode.POST_HOC: 'POST_HOC',
      XaiMode.INTERPRETABLE_MODEL:'INTERPRETABLE_MODEL'
    }
    if(self is None):
      return nameArray
    return nameArray[self]

  def filterComapre(self,other):
    return self.__class__ is other.__class__ and other.value == self.value

  def __str__(self):
    return str(self.value)

class TaskType(Enum):
  CLASSIFICATION = 1
  REGRESSION = 2
  BOTH = 3

  def getName(self=None):
    nameArray= {
      TaskType.CLASSIFICATION: 'CLASSIFICATION',
      TaskType.REGRESSION:'REGRESSION',
      TaskType.BOTH:'BOTH',
    }
    if(self is None):
      return nameArray
    return nameArray[self]

  def filterComapre(self,other):
    if(self.__class__ != other.__class__):
      return False
    if(self.value==3 or other.value==3):
      return True
    return other.value == self.value

  def __str__(self):
    return str(self.value)

class ExplanationScope(Enum):
  GLOBAL = 1
  LOCAL = 2
  BOTH = 3

  def getName(self=None):
    nameArray= {
      ExplanationScope.GLOBAL: 'GLOBAL',
      ExplanationScope.LOCAL:'LOCAL',
      ExplanationScope.BOTH:'BOTH',
    }
    if(self is None):
      return nameArray
    return nameArray[self]

  def filterComapre(self,other):
    if(self.__class__ != other.__class__):
      return False
    if(self.value==3 or other.value==3):
      return True
    return other.value == self.value

  def __str__(self):
    return str(self.value)

class ResultTypes(Enum):
  FI = 1
  VIS = 2
  TEXT = 3
  EXAMPLE = 3

  def getName(self=None):
    nameArray= {
      ResultTypes.FI: 'FI',
      ResultTypes.VIS:'VIS',
      ResultTypes.TEXT:'TEXT',
      ResultTypes.EXAMPLE:'EXAMPLE',
    }
    if(self is None):
      return nameArray
    return nameArray[self]

  def filterComapre(self,other):
    if type(other) == list:
      for elem in other:
        if(self.__class__ == elem.__class__ and elem.value == self.value):
          return True
      return False
    else:
        if(self.__class__ != other.__class__):
          return False
        return other.value == self.value

  def __str__(self):
    return str(self.value)

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