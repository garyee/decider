
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#@title Calculaten and plot correlation
def printCorrMatrix(df):
  corr = df.corr()
  # Generate a mask for the upper triangle
  mask = np.zeros_like(corr, dtype=bool)
  mask[np.triu_indices_from(mask)] = True
  # Set up the matplotlib figure
  f, ax = plt.subplots(figsize=(11, 9))
  # Draw the heatmap with the mask and correct aspect ratio
  myplot=sns.heatmap(corr,
                     mask=mask,
                     cmap= 'coolwarm_r',
                     vmax=.3,
                     annot = True,
                     square=True,
                     xticklabels=True,
                     yticklabels=True,
                     linewidths=.5,
                     cbar_kws={"shrink": .5},
                     ax=ax)
  plt.title('Correlation Matrix for the your Dataset')
  #print(np.absolute(corr.values[np.triu_indices_from(corr.values,1)]).mean())

  

