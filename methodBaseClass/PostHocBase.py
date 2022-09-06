from methodBaseClass.XaiBaseClass import XAIMethod
from abc import abstractmethod
from interpret import show
from utils.enums import ExplanationScope, XaiMode

class PostHocBase(XAIMethod):

    TYPE = XaiMode.POST_HOC

    # @classmethod
    # def __init__(self, df_train, df_y_train):
    #     self.train(df_train, df_y_train)
    #     self.showExplanation(1)
        
    # @classmethod
    # def getMethodStrName(self):
    #     return type(self.CONSTANT).__name__

    # @classmethod
    # def train(self,df_train, df_y_train):
    #     self.CONSTANT.fit(df_train, df_y_train)

    # @classmethod
    # def showExplanation(self,scope,X_test,y_test):
    #     if(scope==ExplanationScope.GLOBAL):
    #         globalExp = self.CONSTANT.explain_global(name=self.getMethodStrName())
    #         show(globalExp)
    #     else:
    #         localExp = self.CONSTANT.explain_local(X_test[:5], y_test[:5], name=self.getMethodStrName())
    #         show(localExp)