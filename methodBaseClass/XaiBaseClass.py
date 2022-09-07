from abc import ABC, abstractmethod
from interpret import show

class XAIMethod(ABC):

  @classmethod
  @property
  @abstractmethod
  def CONSTRAINTS(cls):
      raise NotImplementedError

  def getConstraints(self):
    return self.CONSTRAINTS

  @classmethod
  @property
  @abstractmethod
  def TYPE(cls):
      raise NotImplementedError

  def getXaiMethod(self):
    return self.TYPE

  @classmethod
  @property
  @abstractmethod
  def SCOPE(cls):
      raise NotImplementedError

  def getScope(self):
    return self.SCOPE

  @classmethod
  @property
  @abstractmethod
  def TASKTYPE(cls):
      raise NotImplementedError

  def getTaskType(self):
    return self.TASKTYPE

  @classmethod
  @property
  @abstractmethod
  def NAME(cls):
      raise NotImplementedError

  def getName(self):
    return self.NAME

  @classmethod
  @property
  @abstractmethod
  def RESULTS(cls):
      raise NotImplementedError

  def getResults(self):
    return self.RESULTS