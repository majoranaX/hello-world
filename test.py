# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

#import os
#from utils import DATA_DIR, CHART_DIR
import scipy as sp
import matplotlib.pyplot as plt

sp.random.seed(3)  # to reproduce the data later on

import urllib
#########################################
label_Unit="day"
#label_Unit="week"
#label_Unit="month"
#########################################
#symbol="us.INX"
#symbol="us.DJI"
#symbol="sz399300"
#symbol="usTSM.N"
#symbol="usTSLA.OQ"
#dateUnit="m5"
#dateUnit="day"
#dateUnit="week"
#response=urllib.request.urlopen("http://127.0.0.1/json020cn.php?f=tsv&sym="+symbol+"&u="+dateUnit)


#########################################
#symbol="ADL"
#response=urllib.request.urlopen("http://127.0.0.1/json010.php?f=tsv&sym="+symbol)
##########################################
#symbol="0056"
#symbol="1513"
symbol="6525"
#symbol="2881"
#symbol="2330"
#symbol="6282"
#symbol="00762"
#symbol="00677U"
response=urllib.request.urlopen("http://127.0.0.1/json001.php?u=D&d=240&f=tsv&sym="+symbol)
#########################################
dataTSV = response.read()

symbolFile = "symbol_"+symbol+".txt"

fileTSV= open(symbolFile,"wb+")  
fileTSV.write(dataTSV)
fileTSV.close()

#data = sp.genfromtxt(os.path.join(DATA_DIR, "web_traffic.tsv"), delimiter="\t")

data = sp.genfromtxt(symbolFile, delimiter="\t")

#print(data[:10])
#print(data.shape)

# all examples will have three classes in this file
colors = ['g', 'k', 'b', 'm', 'r','g', 'k', 'b', 'm', 'r']
linestyles = ['-','--','-','--','-', '-.', '-.', '-.', '-.', '-.', '--', ':', '-']

x = data[:, 0]
y = data[:, 5]
#print("Number of invalid entries:", sp.sum(sp.isnan(y)))
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

dataCount = len(data)

if label_Unit=="day":
        dataCount  = int(dataCount/20)
        
if label_Unit=="week":
        dataCount  = int(dataCount/4)
        
# plot input data
def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):

    plt.figure(num=None, figsize=(16, 9))
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title(f"Index over the last {dataCount} month")
    plt.xlabel(label_Unit)
    plt.ylabel("Price")
  #  plt.xticks(
  #      [w * 7 * 24 for w in range(10)], ['week %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = sp.linspace(x.min(), x.max(), 5000)
        for model, style, color in zip(models, linestyles, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=(y.min()*0.999))
    plt.ylim(ymax=(y.max()*1.001))
   # if ymax:
   #     plt.ylim(ymax=ymax)
   # if xmin:
   #     plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname)

# first look at the data
plot_models(x, y, None, symbol+"1400_01_01.png")

# create and plot models
fp1, res1, rank1, sv1, rcond1 = sp.polyfit(x, y, 1, full=True)
print("Model parameters of fp1: %s" % fp1)
print("Error of the model of fp1:", res1)
f1 = sp.poly1d(fp1)

fp2, res2, rank2, sv2, rcond2 = sp.polyfit(x, y, 2, full=True)
print("Model parameters of fp2: %s" % fp2)
print("Error of the model of fp2:", res2)
f2 = sp.poly1d(fp2)
f3 = sp.poly1d(sp.polyfit(x, y, 3))
f4 = sp.poly1d(sp.polyfit(x, y, 4))
f5 = sp.poly1d(sp.polyfit(x, y, 5))
f10 = sp.poly1d(sp.polyfit(x, y, 10))
f12 = sp.poly1d(sp.polyfit(x, y, 12))
f20 = sp.poly1d(sp.polyfit(x, y, 20))
f48 = sp.poly1d(sp.polyfit(x, y, 48))
f60 = sp.poly1d(sp.polyfit(x, y, 60))
f100 = sp.poly1d(sp.polyfit(x, y, 100))

plot_models(x, y, [f1], symbol+"1400_01_02.png")
plot_models(x, y, [f1, f2], symbol+"1400_01_03-1.png")
plot_models(x, y, [f1, f4], symbol+"1400_01_03-2.png")
plot_models(x, y, [f1, f12], symbol+"1400_01_03-3.png")
plot_models(x, y, [f1, f48], symbol+"1400_01_04.png")

# fit and plot a model using the knowledge about inflection point

Du = (int)(x.size/4)
#print(Du)

dev = 4

start = (int)(Du*0)
end   =  (int)(Du*1)
x1 = x[start:end]
y1 = y[start:end]
fd1 = sp.poly1d(sp.polyfit(x1, y1, dev))


start = (int)(Du*1)
end   =  (int)(Du*2)
x2 = x[start:end]
y2 = y[start:end]
fd2 = sp.poly1d(sp.polyfit(x2, y2, dev))
    

start = (int)(Du*2)
end   =  (int)(Du*3)
x3 = x[start:end]
y3 = y[start:end]
fd3 = sp.poly1d(sp.polyfit(x3, y3, dev))
    

start = (int)(Du*3)
end   =  (int)(Du*4)
x4 = x[start:end]
y4 = y[start:end]
fd4 = sp.poly1d(sp.polyfit(x4, y4, dev))


start = (int)(Du*2)
end   =  (int)(Du*4)
x5 = x[start:end]
y5 = y[start:end]
fd5 = sp.poly1d(sp.polyfit(x5, y5, dev))        
    
plot_models(x, y,[f1,fd1,fd2,fd3,fd4], symbol+"1400_01_05.png")
#,
#    mx=sp.linspace(0 * 4 * 5, 15 * 4 * 5, 1000),
#    ymax=5000, xmin=0 * 4 * 5)


# extrapolating into the future
plot_models(
    x, y, [f1, f2, f4, f12, f48],
    symbol+"1400_01_06.png",
    mx=sp.linspace(0 * 4 * 5, 15 * 4 * 5, 1000),
    ymax=5000, xmin=0 * 4 * 5)




fb1 = fd5
fb2 = sp.poly1d(sp.polyfit(x5, y5, 2))
fb3 = sp.poly1d(sp.polyfit(x5, y5, 3))
fb4 = sp.poly1d(sp.polyfit(x5, y5, 4))
fb5 = sp.poly1d(sp.polyfit(x5, y5, 5))
fb10 = sp.poly1d(sp.polyfit(x5, y5, 10))
fb12 = sp.poly1d(sp.polyfit(x5, y5, 12))
fb20 = sp.poly1d(sp.polyfit(x5, y5, 20))
fb48 = sp.poly1d(sp.polyfit(x5, y5, 48))
fb60 = sp.poly1d(sp.polyfit(x5, y5, 60))



plot_models(
    x, y, [fb1, fb2, fb4, fb12, fb48],
    symbol+"1400_01_07.png",
    mx=sp.linspace(0 * 4 * 5, 15 * 4 * 5, 1000),
    ymax=5000, xmin=0 * 4 * 5)


# separating training from testing data
frac = 5/240
split_idx = int(frac * len(x5))
shuffled = sp.random.permutation(list(range(len(x5))))
test = sorted(shuffled[:split_idx])
train = sorted(shuffled[split_idx:])
fbt1 = sp.poly1d(sp.polyfit(x5[train], y5[train], 1))
fbt2 = sp.poly1d(sp.polyfit(x5[train], y5[train], 2))
#print("fbt2(x)= \n%s"%fbt2)
#print("fbt2(x)-100,000= \n%s"%(fbt2-100000))
fbt3 = sp.poly1d(sp.polyfit(x5[train], y5[train], 3))
fbt4 = sp.poly1d(sp.polyfit(x5[train], y5[train], 4))
fbt5 = sp.poly1d(sp.polyfit(x5[train], y5[train], 5))
fbt10 = sp.poly1d(sp.polyfit(x5[train], y5[train], 10))
fbt12 = sp.poly1d(sp.polyfit(x5[train], y5[train], 12))
fbt20 = sp.poly1d(sp.polyfit(x5[train], y5[train], 20))
fbt48 = sp.poly1d(sp.polyfit(x5[train], y5[train], 48))
fbt60 = sp.poly1d(sp.polyfit(x5[train], y5[train], 60))
fbt100 = sp.poly1d(sp.polyfit(x5[train], y5[train], 100))


plot_models(
    x, y, [fbt1,fbt2, fbt4, fbt12,  fbt48],
    symbol+"1400_01_08.png",
    mx=sp.linspace(0 * 4 * 5, 15 * 4 * 5, 1000),
    ymax=5000, xmin=0 * 4 * 5)

