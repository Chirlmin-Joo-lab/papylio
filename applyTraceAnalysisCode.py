# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:50:59 2018

@author: ivoseverins
"""



import os
#os.chdir(r'D:\ivoseverins\SURFdrive\Promotie\Code\Python\traceAnalysis')
os.chdir(r'/Users/ivoseverins/SURFdrive/Promotie/Code/Python')
from traceAnalysis.traceAnalysisCode import *


#mainPath = r'D:\ivoseverins\SURFdrive\Promotie\Code\Python\traceAnalysis\twoColourExampleData\HJ A'

#os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('./traceAnalysis')
mainPath = './twoColourExampleData/HJ A'




exp = Experiment(mainPath)


#exp.histogram(makeFit = True)



#files = os.listdir(e.mainPath)
#filenames = list(set([re.search('hel[0-9]*',file).group() for file in files]))
#
#for filename in filenames: e.addFile(filename,r'.') 



