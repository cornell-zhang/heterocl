"""
This is the black_scholes option pricing model.
A basic model in finance used to decide the price of option.
The model is as follow:

C=S*N(d1)-X*exp(-rT)N(d2)
P=X*exp(-r*T)N(-d2)-S*N(-d1)

where:
d1=(ln(S/X)+(r+sigma^2/2)T)/(sigma*sqrt(T))
d2=d1-sigma*sqrt(T)

C:the output: Bullish price
P:the output: Down price
X:Option execution Price
S:Current price of financial assets
r:Risk-free continuous compound interest rate
sigma:Annual volatility rate of stock continuous compounding rate
N():Cumulative probability distribution function of unit normal distribution

More information could be found here:https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model

=================================
"""
#encoding:utf-8
import numpy as np
import heterocl as hcl
import math
hcl.config.init_dtype = "float32"
#----------------------------------------
#The input and output parameters.
#xdiv1 is d1;xdiv2 is d2
#d1Local is N(d1); d2local is N(d2)
Loop = 1
X = hcl.placeholder((1,))
T = hcl.placeholder((1,))
S = hcl.placeholder((1,))
r = hcl.placeholder((1,))
sigma = hcl.placeholder((1,))
C = hcl.placeholder((1,))
P = hcl.placeholder((1,))
xdiv1 = hcl.placeholder((1,))
xdiv2 = hcl.placeholder((1,))
d1Local = hcl.placeholder((1,))
d2Local = hcl.placeholder((1,))
#------------------------------------------#
with hcl.stage() as s:
	xlogterm = hcl.scalar(0)
	xlogterm[0] = hcl.log(S[0]/X[0])
	xpowerterm = hcl.scalar(0)
	xpowerterm[0] = 0.5*sigma[0]*sigma[0]
	xnum = hcl.scalar(0)
	xnum[0] = xlogterm[0]+(r[0]+xpowerterm[0])*T[0]
	xsqrtterm = hcl.scalar(0)
	xsqrtterm[0] = hcl.sqrt(T[0])
	xden = hcl.scalar(0)
	xden[0] = sigma[0]*xsqrtterm[0]
	#xdiv1 = hcl.scalar(0)
	xdiv1[0] = xnum[0]/xden[0]
	#xdiv2 = hcl.scalar(0)
	xdiv2[0] = xdiv1[0]-xden[0]
	futurevaluex = hcl.scalar(0)
	futurevaluex[0] = X[0]*hcl.exp(-r[0]*T[0])
	#--------------------------------------------------#
	#Calculate N(d1), also N(-d1)=1 - N(d1)
	d1NPrimeofX = hcl.scalar(0)
	d1NPrimeofX[0] = hcl.exp(-(xdiv1[0]*xdiv1[0])*0.5)*0.39894228040143270286
	d1K2 = hcl.scalar(0)
	d1K2[0] = 1/((xdiv1[0]*0.2316419)+1.0)
	d1K2_2 = hcl.scalar(0)
	d1K2_2[0] = d1K2[0]*d1K2[0]
	d1K2_3 = hcl.scalar(0)
	d1K2_3[0] = d1K2_2[0] * d1K2[0]
	d1K2_4 = hcl.scalar(0)
	d1K2_4[0] = d1K2_3[0] * d1K2[0]
	d1K2_5 = hcl.scalar(0)
	d1K2_5[0] = d1K2_4[0] * d1K2[0]
	
	d1Local_10 = hcl.scalar(0)
	d1Local_10[0] = d1K2[0] * 0.319381530
	d1Local_20 = hcl.scalar(0)
	d1Local_20[0] = d1K2_2[0] * -0.356563782
	d1Local_30 = hcl.scalar(0)
	d1Local_30[0] = d1K2_3[0] * 1.781477937
	d1Local_31 = hcl.scalar(0)
	d1Local_31[0] = d1K2_4[0] * -1.821255978
	d1Local_32 = hcl.scalar(0)
	d1Local_32[0] = d1K2_5[0] * 1.330274429
	
	d1Local_21 = hcl.scalar(0)
	d1Local_21[0] = d1Local_20[0] + d1Local_30[0]
	d1Local_22 = hcl.scalar(0)
	d1Local_22[0] = d1Local_21[0] + d1Local_31[0]
	d1Local_23 = hcl.scalar(0)
	d1Local_23[0] = d1Local_22[0] + d1Local_32[0]
	d1Local_1 = hcl.scalar(0)
	d1Local_1[0] = d1Local_23[0] + d1Local_10[0]

	d1Local0 = hcl.scalar(0)
	d1Local0[0] = d1Local_1[0] * d1NPrimeofX[0]
	
	#d1Local  = hcl.scalar(0)
	d1Local[0]  = -d1Local0[0] + 1.0
	#---------------------------------------------#
	#Calculate N(d2), also N(-d2)=1 - N(d1)
	d2NPrimeofX = hcl.scalar(0)
	#1/sqrt(2*pi)=0.39894228040143270286
	d2NPrimeofX[0] = (hcl.exp(-(xdiv2[0]*xdiv2[0])*0.5))*0.39894228040143270286 
	d2K2 = hcl.scalar(0)
	d2K2[0] = 1/((xdiv2[0]*0.2316419)+1.0)
	d2K2_2 = hcl.scalar(0)
	d2K2_2[0] = d2K2[0]*d2K2[0]
	d2K2_3 = hcl.scalar(0)
	d2K2_3[0] = d2K2_2[0] * d2K2[0]
	d2K2_4 = hcl.scalar(0)
	d2K2_4[0] = d2K2_3[0] * d2K2[0]
	d2K2_5 = hcl.scalar(0)
	d2K2_5[0] = d2K2_4[0] * d2K2[0]
	
	d2Local_10 = hcl.scalar(0)
	d2Local_10[0] = d2K2[0] * 0.319381530
	d2Local_20 = hcl.scalar(0)
	d2Local_20[0] = d2K2_2[0] * -0.356563782
	d2Local_30 = hcl.scalar(0)
	d2Local_30[0] = d2K2_3[0] * 1.781477937
	d2Local_31 = hcl.scalar(0)
	d2Local_31[0] = d2K2_4[0] * -1.821255978
	d2Local_32 = hcl.scalar(0)
	d2Local_32[0] = d2K2_5[0] * 1.330274429
	
	d2Local_21 = hcl.scalar(0)
	d2Local_21[0] = d2Local_20[0] + d2Local_30[0]
	d2Local_22 = hcl.scalar(0)
	d2Local_22[0] = d2Local_21[0] + d2Local_31[0]
	d2Local_23 = hcl.scalar(0)
	d2Local_23[0] = d2Local_22[0] + d2Local_32[0]
	d2Local_1 = hcl.scalar(0)
	d2Local_1[0] = d2Local_23[0] + d2Local_10[0]

	d2Local0 = hcl.scalar(0)
	d2Local0[0] = d2Local_1[0] * d2NPrimeofX[0]
	
	#d2Local  = hcl.scalar(0)
	d2Local[0]  = -d2Local0[0] + 1.0
	#---------------------------------------------------#
	#Calculate C and P
	C[0] = (S[0]*d1Local[0]) - (futurevaluex[0]*d2Local[0])
	P[0] = (futurevaluex[0]*d2Local0[0]) - (S[0]*d1Local0[0])
	#------------------------------------------------------------#
	
o = hcl.create_schedule(s)
print hcl.lower(o, [X,T,S,r,sigma,C,P,xdiv1,xdiv2,d1Local,d2Local])
f = hcl.build(o, [X,T,S,r,sigma,C,P,xdiv1,xdiv2,d1Local,d2Local])

#---------------------------------------------------------------------#
'''
#This is the same algorithm in Python
# Give an example set of parameters
X0 = 165
T0 = 0.0959
S0 = 164
r0 = 0.0521
sigma0 = 0.29
C0 = 0
P0 = 0
xdiv10 = 0
xdiv20 = 0
d1Local0 = 0
d2Local0 = 0

xlogterm0 = math.log(0.9939393939393939)
xpowerterm0 = 0.5*sigma0*sigma0
xnum0 = xlogterm0+(r0+xpowerterm0)*T0
xsqrtterm0 = math.sqrt(T0)
xden0 = sigma0*xsqrtterm0
xdiv10 = xnum0/xden0
xdiv20 = xdiv10 - xden0
futurevaluex0 = X0*(math.exp(-r0*T0))

#Calculate N(d1), also N(-d1)=1-N(d1)
d1NPrimeofX0 = hcl.exp(-(xdiv10*xdiv10)*0.5)*0.39894228040143270286
d1K20 = 1/((xdiv10*0.2316419)+1.0)
d1K2_20 = d1K20*d1K20
d1K2_30 = d1K2_20 * d1K20
d1K2_40 = d1K2_30 * d1K20
d1K2_50 = d1K2_40 * d1K20

d1Local_100 = d1K20 * 0.319381530
d1Local_200 = d1K2_20 * -0.356563782
d1Local_300 = d1K2_30 * 1.781477937
d1Local_310 = d1K2_40 * -1.821255978
d1Local_320 = d1K2_50 * 1.330274429
	

d1Local_210 = d1Local_200 + d1Local_300
d1Local_220 = d1Local_210 + d1Local_310
d1Local_230 = d1Local_220 + d1Local_320
d1Local_10 = d1Local_230 + d1Local_100

d1Local00 = d1Local_10 * d1NPrimeofX0
d1Local0  = -d1Local00 + 1.0

#Calculate N(d2), also N(-d2)=1 - N(d1)
#1/sqrt(2*pi)=0.39894228040143270286
d2NPrimeofX0 = (hcl.exp(-(xdiv20*xdiv20)*0.5))*0.39894228040143270286
d2K20 = 1/((xdiv20*0.2316419)+1.0)
d2K2_20 = d2K20*d2K20
d2K2_30 = d2K2_20 * d2K20
d2K2_40 = d2K2_30 * d2K20
d2K2_50 = d2K2_40 * d2K20
	
d2Local_100 = d2K20 * 0.319381530
d2Local_200 = d2K2_20 * -0.356563782
d2Local_300 = d2K2_30 * 1.781477937
d2Local_310 = d2K2_40 * -1.821255978
d2Local_320 = d2K2_50 * 1.330274429
	
d2Local_210 = d2Local_200 + d2Local_300
d2Local_220 = d2Local_210 + d2Local_310
d2Local_230 = d2Local_220 + d2Local_320
d2Local_10 = d2Local_230 + d2Local_100

d2Local00 = d2Local_10 * d2NPrimeofX0
d2Local0 = -d2Local00 + 1.0

C0 = (S0*d1Local0) - (futurevaluex0*d2Local0)
P0 = (futurevaluex0*d2Local00) - (S0*d1Local00)
'''
#--------------------------------------------------#
hcl_X = hcl.asarray([165], dtype = hcl.Float())
hcl_T = hcl.asarray([0.0959], dtype = hcl.Float())
hcl_S = hcl.asarray([164], dtype = hcl.Float())
hcl_r = hcl.asarray([0.0521], dtype = hcl.Float())
hcl_sigma = hcl.asarray([0.29], dtype = hcl.Float())
hcl_C = hcl.asarray([0], dtype = hcl.Float())
hcl_P = hcl.asarray([0], dtype = hcl.Float())
hcl_xdiv1 = hcl.asarray([0], dtype = hcl.Float())
hcl_xdiv2 = hcl.asarray([0], dtype = hcl.Float())
hcl_d1Local = hcl.asarray([0], dtype = hcl.Float())
hcl_d2Local = hcl.asarray([0], dtype = hcl.Float())

f(hcl_X,hcl_T,hcl_S,hcl_r,hcl_sigma,hcl_C,hcl_P,hcl_xdiv1,hcl_xdiv2,hcl_d1Local,hcl_d2Local)



print("-------This is the result of heterocl-------------")
print("Bullish price:")
print hcl_C
print("Down price:")
print hcl_P
print("value of d1:")
print hcl_xdiv1
print("value of d2:")
print hcl_xdiv2
print("value of N(d1):")
print hcl_d1Local
print("value of N(d2):")
print hcl_d2Local

'''
print("-------This is the result of Python------------")
print("Bullish price:")
print C0
print("Down price:")
print P0
print("value of d1:")
print xdiv10
print("value of d2:")
print xdiv20
print("value of N(d1):")
print d1Local0
print("value of N(d2):")
print d2Local0
'''
	
	
	
	

