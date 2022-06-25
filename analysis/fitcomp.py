import matplotlib as mpl
mpl.rc('figure',facecolor='white')
mpl.rc('lines', markersize = 1.6 )
mpl.rc('lines', markeredgewidth = 0.0 )
#mpl.rc('font', **{'family':'sans-serif','sans-serif':['Arial']})
import matplotlib.pyplot as plt
import time
import matplotlib.dates as dt
import numpy as np
import datetime
import ROOT
from scipy import integrate
from scipy import interpolate
from scipy.special import erfc
from scipy.optimize import curve_fit
from lmfit.models import SkewedVoigtModel
from lmfit.models import ExponentialGaussianModel
from lmfit.models import SkewedGaussianModel
from lmfit import Model

#amplitude:  33.4567195 +/- 0.65920093 (1.97%) (init = 32.94747)
#center:     80.4117231 +/- 0.06320307 (0.08%) (init = 82)
#sigma:      1.22147643 +/- 0.06437482 (5.27%) (init = 1.2)
#skew:       1.39659135 +/- 0.20106826 (14.40%) (init = 0)
#gamma:      1.22147643 +/- 0.06437482 (5.27%) == 'sigma'

#p0                        =        41.22
#p1                        =      41.4702
#p2                        =       0.5944
#p3                        =        395.3
#p4                        =        3.598
#p5                        =      1.10272
#p6                        =      80.4117
#p7                        =      -390916
#p8                        =         81.9


pkmodel = SkewedVoigtModel()
catmodel = ExponentialGaussianModel()
pars = pkmodel.make_params()
catpars = catmodel.make_params()
err = 0.0
sqrt2 = np.sqrt(2.0)

def diff_func(x,cat,an,offst,thold,tcrise,tarise,center,gamma,skew):
    global err
    y = cat*np.exp(-((-x+10.0+tcrise**2/thold)/(sqrt2*tcrise))**2)*np.exp( -(x-10.0-tcrise**2/(2*thold))/thold )
    y = y + (-1.0/thold)*0.5*cat*np.exp( -(x-10.0-tcrise**2/(2*thold))/thold )*erfc((-x+10.0+tcrise**2/thold)/(sqrt2*tcrise))
    pars['amplitude'].value = an
    pars['sigma'].value = tarise
    pars['center'].value = center
    pars['gamma'].value = gamma
    pars['skew'].value = skew
    integrand = lambda xi : pkmodel.eval( pars, x=xi )
    norm = integrate.quad( integrand, -np.inf , np.inf )[0] 
    svint = np.array([ integrate.quad( integrand, -np.inf,xi )[0] for xi in x ])/norm
    y = y - pkmodel.eval( pars, x=x )*np.exp( -(x-81.9-tarise**2/(2*thold))/thold )
    y = y + (1.0/thold)*an*svint*np.exp( -(x-81.9)/thold )
    return y

def fitter_func(x,cat,an,offst,thold,tcrise,tarise):
    global err
    x_beg = x[x<10.0]
    x_mid = x[(x>=10.0)*(x<81.9)]
    x_end = x[x>=81.9]
    y_beg = 0.5*cat*erfc((-x_beg+10.0)/tcrise) - 0.5*an*erfc((-x_beg+81.9)/tarise)
    y_mid = 0.5*cat*erfc((-x_mid+10.0)/tcrise)*np.exp(-(x_mid-10.0)/thold) - 0.5*an*erfc((-x_mid+81.9)/tarise)
    y_end = 0.5*cat*erfc((-x_end+10.0)/tcrise)*np.exp(-(x_end-10.0)/thold) - 0.5*an*erfc((-x_end+81.9)/tarise)*np.exp(-(x_end-81.9)/thold)
    y = np.concatenate((y_beg,y_mid,y_end),axis=None)
    y = y + offst
    return y

def theory_func(x,cat,an,offst,thold,tcrise,tarise):
    global err
    y = 0.5*cat*erfc((-x+10.0+tcrise**2/thold)/(sqrt2*tcrise))*np.exp( -(x-10.0-tcrise**2/(2*thold))/thold )
    y = y - 0.5*an*erfc((-x+81.9+tarise**2/thold)/(sqrt2*tarise))*np.exp( -(x-81.9-tarise**2/(2*thold))/thold )
    y = y + offst
    return y

def thesis_func(x,cat,an,offst,thold,tcrise,tarise):
    global err
    y = 0.5*cat*erfc((-x+10.0)/(tcrise))*np.exp( -(x-10.0)/thold )
    y = y - 0.5*an*erfc((-x+81.9)/(tarise))*np.exp( -(x-81.9)/thold )
    y = y + offst
    return y

def svi(x,tarise,center,gamma,skew) :
    pars['sigma'].value = tarise
    pars['center'].value = center
    pars['gamma'].value = gamma
    pars['skew'].value = skew
    integrand = lambda xi : pkmodel.eval( pars, x=xi )
    norm = integrate.quad( integrand, -np.inf , np.inf )[0] 
    y = np.array([ integrate.quad( integrand, -np.inf,xi )[0] for xi in x ])/norm
    return y

def double_diff(x,cat,an,tcrise,cent_c,gam_c,tarise,cent_a,gam_a,skew_a):
    catpars['amplitude'].value = cat
    catpars['sigma'].value = tcrise
    catpars['center'].value = cent_c
    catpars['gamma'].value = gam_c
    y = catmodel.eval( catpars, x=x )#*np.exp( -(x-10.0)/thold )
    pars['amplitude'].value = an
    pars['sigma'].value = tarise
    pars['center'].value = cent_a
    pars['gamma'].value = gam_a
    pars['skew'].value = skew_a
    y = y - pkmodel.eval( pars, x=x)#*np.exp( -(x-81.9)/thold )
    return y
 
def smeared_func(x,cat,an,offst,thold,tcrise,tarise,center,gamma,skew):
    global err
    y = 0.5*cat*erfc((-x+10.0+tcrise**2/thold)/(sqrt2*tcrise))*np.exp( -(x-10.0-tcrise**2/(2*thold))/thold )
    pars['amplitude'].value = an
    pars['sigma'].value = tarise
    pars['center'].value = center
    pars['gamma'].value = gamma
    pars['skew'].value = skew
    integrand = lambda xi : pkmodel.eval( pars, x=xi )
    norm = integrate.quad( integrand, -np.inf , np.inf )[0] 
    svint = np.array([ integrate.quad( integrand, -np.inf,xi )[0] for xi in x ])/norm
    y = y - an*svint*np.exp( -(x-81.9)/thold )
    y = y + offst
    return y

def extra_smeared(x,cat,an,tcrise,cent_c,gam_c,tarise,cent_a,gam_a,skew_a,offst):
    catpars['amplitude'].value = cat
    catpars['sigma'].value = tcrise
    catpars['center'].value = cent_c
    catpars['gamma'].value = gam_c
    #tfine = np.arange(x[-1]-1000.0,x[-1],0.08)
    integrand_c = catmodel.eval(catpars, x=x )
    integral_c = integrate.cumulative_trapezoid( integrand_c, x) 
    integral_c = np.append( integral_c, integral_c[-1] )
    y = integral_c*np.exp( -(x-10.0)/395.3 )
    pars['amplitude'].value = an
    pars['sigma'].value = tarise
    pars['center'].value = cent_a
    pars['gamma'].value = gam_a
    pars['skew'].value = skew_a
    integrand_a = pkmodel.eval(pars, x=x )
    integral_a = integrate.cumulative_trapezoid( integrand_a, x) 
    integral_a = np.append( integral_a, integral_a[-1] )
    y = y - integral_a*np.exp( -(x-81.9)/395.3 )
    y = y + offst
    return y


t = []
volt = []
wf = []
#with open('sig_plus_bkg1.dat') as ff:
with open('testwaveform_0956.dat') as ff:
    for line in ff:
        try :
            t.append( 1000000.0*float(line.split(',')[0]) )
            volt.append( 1000.0*float(line.split(',')[1]) )
        except ValueError:
            continue
#with open('bkg.dat') as ff:
import glob
files = glob.glob('bkg.dat')
with open(files[0]) as ff:
    for ln, line in enumerate(ff):
        try :
            volt[ln] = 1.0*volt[ln]# - 1000.0*float(line.split(',')[1])
        except Exception:
            continue
#with open('rawsynth14bit.dat') as ff:
#    for line in ff:
#        try :
#            wfm = [ float(u) for u in line.split(',') ]
#        except ValueError:
#            continue
#with open('/home/kolo/nEXO/nEXOskyline/XPM_analysis/data/FitterLast2.csv') as ff:
#with open('sample_wfm.csv') as ff:
#    for line in ff:
#        try :
#            t.append( 1000000.0*float(line.split(',')[0]) )
#            volt.append( 1000.0*float(line.split(',')[1]) )
#        except ValueError:
#            continue

with open('raw_sig.dat') as ff:
    for line in ff:
        try :
            wfm = [ float(u) for u in line.split(',') ]
        except ValueError:
            continue
with open('raw_bkg.dat') as ff:
    for line in ff:
        try :
            raw_bkg = [ float(u) for u in line.split(',') ]
        except ValueError:
            continue
#wfmpre = '2;16;ASC;RP;MSB;500;"Ch1, AC coupling, 2.0E-2 V/div, 4.0E-5 s/div, 500 points, Average mode";Y;8.0E-7;0;-1.2E-4;"s";3.125E-6;0.0E0;-1.3824E4;"V"\n' #16 bit
wfmpre = '1;8;ASC;RP;MSB;500;"Ch1, AC coupling, 2.0E-2 V/div, 4.0E-5 s/div, 500 points, Average mode";Y;8.0E-7;0;-1.2E-4;"s";8.0E-4;0.0E0;-5.4E1;"V"'
t = [ 1.0e6*(float(wfmpre.split(';')[8])*float(i)+float(wfmpre.split(';')[10])) for i in range(0,500) ]
volt = np.array([ 1.0e3*(( dl - float(wfmpre.split(';')[14]) )*float(wfmpre.split(';')[12]) - float(wfmpre.split(';')[13])) for dl in wfm ])
#volt = volt + 44.6*2
bkg = [ 1.0e3*(( dl - float(wfmpre.split(';')[14]) )*float(wfmpre.split(';')[12]) - float(wfmpre.split(';')[13])) for dl in raw_bkg ]
volt = np.array([ v[0] - v[1] for v in zip( volt , bkg ) ])

volt = np.array(volt)
t = np.array(t)
vprime = np.gradient(volt)/np.gradient(t)
#baseline = np.concatenate((volt[0:150],volt[0:150],volt[0:150],volt[0:50]),axis=None)
#baseline = baseline - np.mean(baseline)

#p_i = [66.22,65.4702,41.5944,395.3,3.598,1.10272,80.4117,-390916.0,81.9]
#p_i = [65.669502,66.09416,41.967726,395.3,3.598,1.0884104,81.320328,1.80825,0.38634939]
p_i = [37.873185672822736,40.81570955383812,10.0,3.598,0.980325759727434,81.9,1.80825,0.8,0.9,0.2]
#p_i = [66.22,65.4702,41.5944,395.3,1.0,2.9,80.4117,-390916.0,81.9]
#p_i = [41.22,41.4702,0.5944,395.3,3.598,1.0884104,81.320328,1.80825,0.38634939]
#svi_norm = svi(t,p_i[5],p_i[6],p_i[7],p_i[8]) 
#svi_norm = interpolate.interp1d( t, svi_norm )

def fast_smeared_func(x,cat,an,offst,thold,tcrise,tarise,center,gamma,skew):
    global err
    y = 0.5*cat*erfc((-x+10.0+tcrise**2/thold)/(sqrt2*tcrise))*np.exp( -(x-10.0-tcrise**2/(2*thold))/thold )
    y = y-(an*svi_norm(x))*np.exp( -(x-center)/thold ) + offst
    return y

import random
#randidx = random.randint(0,500)
#baseline = [ baseline[ (randidx+jj)%500] for jj in range(0,500) ]

import csv
#with open('/home/kolo/nEXO/nEXOskyline/XPM_analysis/data/noise_sim.csv','w',newline='') as fout:
#    noiser = csv.writer( fout , delimiter = ',' )
#    for ms,mV in zip(t/1000.0,baseline) :
#        noiser.writerow([ms,mV])


#wavmodel = Model(smeared_func,nan_policy='raise')
wavmodel = Model(extra_smeared,nan_policy='raise')
#wavmodel = Model(fitter_func,nan_policy='raise')
wavparams = wavmodel.make_params()

wavparams['cat'].value = p_i[0]
wavparams['cat'].vary = True
wavparams['an'].value = p_i[1]
wavparams['an'].vary = True
wavparams['cent_c'].value = p_i[2]
wavparams['cent_c'].vary = True
#wavparams['thold'].value = p_i[3]
#wavparams['thold'].vary = False
wavparams['tcrise'].value = p_i[3]
wavparams['tcrise'].vary = True
wavparams['tarise'].value = p_i[4]
wavparams['tarise'].vary = True
wavparams['cent_a'].value = p_i[5]
wavparams['cent_a'].vary = True
wavparams['gam_a'].value = p_i[6]
wavparams['gam_a'].vary = True
wavparams['skew_a'].value = p_i[7]
wavparams['skew_a'].vary = True
wavparams['gam_c'].value = p_i[8]
wavparams['gam_c'].vary = True
wavparams['offst'].value = p_i[9]
wavparams['offst'].vary = True

result = wavmodel.fit(volt[t<150],wavparams,x=t[t<150])
#plt.plot(t,baseline)
#plt.plot(t,volt)
#plt.plot(t,vprime)
b = result.best_values
#b = wavparams
#waveform = Model(double_diff,nan_policy='raise')
plt.plot(t,volt)
#plt.plot(t,waveform.eval(x=t,an=b['an'],cat=b['cat'],cent_c=b['cent_c'],thold=b['thold'],tcrise=b['tcrise'],tarise=b['tarise'],cent_a=b['cent_a'],gam_a=b['gam_a'],skew_a=b['skew_a'],gam_c=b['gam_c'],skew_c=b['skew_c']))

tfine = np.arange(t[0],t[-1]+0.8,(t[1]-t[0])/10.0)
plt.plot(tfine,wavmodel.eval(x=tfine,an=b['an'],cat=b['cat'],cent_c=b['cent_c'],tcrise=b['tcrise'],tarise=b['tarise'],cent_a=b['cent_a'],gam_a=b['gam_a'],gam_c=b['gam_c'],skew_a=b['skew_a'],offst=b['offst']))
#plt.figure()
#plt.plot(t,volt)

#catpars['amplitude'].value = b['cat']
#catpars['sigma'].value = b['tcrise']
#catpars['center'].value = b['cent_c']
#catpars['gamma'].value = b['gam_c']
#integrand_c = lambda ti: catmodel.eval(catpars, x=ti )

#yfit = np.array([ integrate.quad( integrand_c, -1000.0,xi) for xi in tfine ])
#integrand_c = catmodel.eval(catpars, x=tfine )
#yfit = integrate.cumulative_trapezoid( integrand_c, tfine)
#tfine = tfine[0:-1]
#yfit = yfit*np.exp(-(tfine-10.0)/395.3)
#pars['amplitude'].value = b['an']
#pars['sigma'].value = b['tarise']
#pars['center'].value = b['cent_a']
#pars['gamma'].value = b['gam_a']
#pars['skew'].value = b['skew_a']
#integrand_a = lambda ti: pkmodel.eval(pars, x=ti )
#yfit = yfit - np.array([ integrate.quad( integrand_a, -np.inf,xi)[0] for xi in tfine ])*np.exp(-(tfine-81.9)/395.3)
#plt.plot(tfine,yfit+1.75)
#plt.plot(t,pkmodel.eval(x=t,amplitude=p_i[1],sigma=51.6,center=p_i[6],gamma=5.0,skew=p_i[8]))
#vc = 0.5*p_i[0]*erfc((-t+10.0+p_i[4]**2/p_i[3])/(sqrt2*p_i[4]))*np.exp( -(t-10.0-p_i[4]**2/(2*p_i[3]))/p_i[3] )
#downstroke =(vc-volt+p_i[2])*np.exp((t-p_i[6])/p_i[3]) 
#dp = np.diff(downstroke)
#pars['amplitude'].value = p_i[1]
#pars['sigma'].value = p_i[5]
#pars['center'].value = p_i[6]
#pars['gamma'].value = p_i[7]
#pars['skew'].value = p_i[8]
#result = pkmodel.fit(dp[200:499],pars,x=t[200:499])
#plt.plot(t[200:499],dp[200:499])
#plt.plot(t[200:499],result.best_fit)
#plt.plot(t,wavmodel.eval(x=t,an=p_i[0],cat=p_i[1],offst=p_i[2],thold=p_i[3],tcrise=p_i[4],tarise=p_i[5],center=p_i[6],gamma=p_i[7],skew=p_i[8]))
#print(result.params)
#print('Lifetime ' + str((81.9-10.0)/np.log(result.params['cat']/result.params['an'])) )
#print('Lifetime ' + str((81.9-10.0)/np.log(wavparams['cat']/wavparams['an'])) )
plt.show()
