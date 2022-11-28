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
  LOW = -1
  MEDIUM = 0
  HIGH = 3

  def rank(self,numValue):
      return self.value
      
  def __str__(self):
    return str(self.value)


class Complexity(Enum):
  HIGH = -3
  MEDIUM = 0
  LOW = 3

  def rank(self,numValue):
    return self.value
    
  def __str__(self):
    return str(self.value)

class Interactivity(Enum):
  NO = 1
  MANUALLY = 2
  YES = 3

  def rank(self,numValue):
    if(self.value==Interactivity.YES.value):
        return 3
    if(self.value==Interactivity.NO.value):
      if(numValue>1):
        return -10
      if(numValue>0.5):
        return -3
    if(self.value==Interactivity.MANUALLY.value):
      if(numValue>1):
        return -2
      if(numValue>0.5):
        return 1
    return 1

  def __str__(self):
    return str(self.value)

class Heterogeneity(Enum):
  CATEGORICAL = 1
  NUMMERICAL = 2
  BOTH = 3

  def rank(self,numValue):
    if(self.value==Heterogeneity.BOTH.value):
      return 2
    if(self.value==Heterogeneity.NUMMERICAL.value):
      if(numValue>0.5):
        return -1
    if(self.value==Heterogeneity.CATEGORICAL.value):
      if(numValue<0.5):
        return -1
    return 1

  def __str__(self):
    return str(self.value)

class Monotonicity(Enum):
  NO = 1
  YES = 2
  BOTH = 3

  def rank(self,numValue):
    if(self.value==Monotonicity.BOTH.value):
      return 2
    if(self.value==Monotonicity.YES.value):
      if(numValue<0.8):
        return -1
    if(self.value==Monotonicity.NO.value):
      if(numValue>0.8):
        return -1
    return 1

  def __str__(self):
    return str(self.value)

class Linearity(Enum):
  NO = 1
  YES = 2
  BOTH = 3

  def rank(self,numValue):
    if(self.value==Linearity.BOTH.value):
      return 2
    if(self.value==Linearity.YES.value):
      if(numValue<0.4):
        return -2
    if(self.value==Linearity.NO.value):
      if(numValue>0.4):
        return -3
    return 1

  def __str__(self):
    return str(self.value)

#The effect Correlation has 0=correlated
class Correlation(Enum):
  VERYBAD = -3
  BAD = 0
  NONE = 3

  def rank(self,numValue):
    if(self.value==Correlation.NONE.value):
      return 3
    if(self.value==Correlation.BAD.value):
      if(numValue<0.5):
        return -2
    if(self.value==Correlation.VERYBAD.value):
      if(numValue<0.5):
        return -5
    return 1

  def __str__(self):
    return str(self.value)

#>10 is mc
class Multicollinearity(Enum):
  VERYBAD = 1
  BAD = 2
  NONE = 3

  def rank(self,numValue):
    if(self.value==Multicollinearity.NONE.value):
      return 2
    if(self.value==Multicollinearity.BAD.value):
      if(numValue>10):
        return -2
    if(self.value==Multicollinearity.VERYBAD.value):
      if(numValue>5):
        return -3
    return 1

  def __str__(self):
    return str(self.value)

# if bad 447 => 1 & >447 -> eponentially bad
class ColumCount(Enum):
  BAD = 1
  NONE = 3

  def rank(self,numValue):
    if(self.value==ColumCount.NONE.value):
      return 2
    if(self.value==ColumCount.BAD.value):
      colCountRatio=int((numValue/100)**2)
      if(numValue>20):
        return (-1*colCountRatio)
    return 1

  def __str__(self):
    return str(self.value)
