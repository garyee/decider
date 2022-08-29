from abc import ABC, abstractmethod
from interpret import show

class XAIMethod(ABC):

  @classmethod
  @property
  @abstractmethod
  def CONSTANT(cls):
      raise NotImplementedError
  
  @classmethod
  def __init__(self, df_train, df_y_train):
        self.train(df_train, df_y_train)
        self.showExplanation(1)
        
  @classmethod
  def getMethodStrName(self):
    return type(self.CONSTANT).__name__

  @classmethod
  def train(self,df_train, df_y_train):
    self.CONSTANT.fit(df_train, df_y_train)

  @classmethod
  def showExplanation(self,glob_or_local):
    if(glob_or_local==1):
      globalExp = self.CONSTANT.explain_global(name=self.getMethodStrName())
      show(globalExp)
    else:
      localExp = self.CONSTANT.explain_local(X_test[:5], y_test[:5], name=self.getMethodStrName())
      show(localExp)
