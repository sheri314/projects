import random
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sys

prices = []
TRIALS = 40
X_ = []

f = open("prices.txt")
for line in f: # extracts data from prices.txt file them and stores into an array called prices
    v = float(line.strip())
    prices.append(v)

N = len(prices)

def sample(data): # returns new array of random samples with replacement
    newArray = []
    for i in range(0, N):
        randomNum = random.randint(0,N-1)
        newArray.append(data[randomNum])
    return newArray

for i in range(0,TRIALS): # adds in sample means of say, 40 trials
    X_.append(np.mean(sample(prices)))

X_.sort()

lowerIndex = int(TRIALS * 0.025)
upperIndex = int(TRIALS * 0.975)-1

print 'left index is', X_[lowerIndex]
print 'right index is', X_[upperIndex]

mean = np.mean(X_)
stddev = math.sqrt(np.var(X_))

clt_left = np.mean(prices) - (1.96 * (np.std(prices)/np.sqrt(N)))
clt_right = np.mean(prices) + (1.96 * (np.std(prices)/np.sqrt(N)))

#print 'clt_left is', clt_left
#print 'clt_right is', clt_right

fig = plt.figure()
ax = fig.add_subplot(111)

x = np.arange(1.05, 1.25, 0.001)
plt.axis([1.10, 1.201, 0, 30])
x = np.arange(1.05, 1.25, 0.001)
y = stats.norm.pdf(x, mean, stddev) # WHAT ARE MEAN AND STDDEV?
plt.plot(x, y, color='red')

ci_x = np.arange(clt_left, clt_right, 0.001)
ci_y = stats.norm.pdf(ci_x, mean, stddev) # or use stats.norm.pdf()
plt.fill_between(ci_x,ci_y,color="#F8ECE0")

plt.scatter(X_[lowerIndex], 0, marker = 'D')
plt.scatter(X_[upperIndex], 0, marker = 'D')

fancy = False
if len(sys.argv) > 1:
    TRIALS = int(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == '-fancy':
        fancy = True
        plt.text(.02,.95, '$TRIALS = %d$' % TRIALS, transform = ax.transAxes)
        plt.text(.02,.9, '$mean(prices)$ = %f' % np.mean(prices), transform = ax.transAxes)
        plt.text(.02,.85, '$mean(\\overline{X})$ = %f' % np.mean(X_), transform = ax.transAxes)
        plt.text(.02,.80, '$stddev(\\overline{X})$ = %f' % np.std(X_,ddof=1), transform = ax.transAxes)
        plt.text(.02,.75, '95%% CI = ($%1.2f,\\ %1.2f$)' % (clt_left, clt_right), transform = ax.transAxes)
        plt.text(1.135, 11.5, "Expected", fontsize=16)
        plt.text(1.135, 10, "95% CI $\\mu \\pm 1.96\\sigma$", fontsize=16)
        plt.title("95% Confidence Intervals: $\\mu \\pm 1.96\\sigma$", fontsize=16)

        ax.annotate("Empirical 95% CI",
                    xy=(X_[lowerIndex], 0),
                    xycoords="data",
                    xytext=(1.13,4), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),
                    fontsize=16)
plt.savefig('bootstrap-'+str(TRIALS)+('-basic' if not fancy else '')+'.pdf', format="pdf")

plt.show()