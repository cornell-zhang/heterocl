"""
This is the Gaussian discriminant analysis written in Heterocl .

Guasian discriminant analysis is a machine learning algorithm used in classification
x0 is the points of the first category, the number is R0. x1 is the points of the second category, the number is R1.

Using the points we can calculate several parameters.

phi is the proportion of the second category.
u0 is the mean value of the first category.
u1 is the mean value of the second category.
sigma is the covariance matrix 

See more information about gda here: http://cs229.stanford.edu/notes/cs229-notes2.pdf
=================================
"""
#encoding:utf-8
import numpy as np
import heterocl as hcl 
hcl.config.init_dtype = "float32"

R0 = 524
R1 = 524
D = 96

x0 = hcl.placeholder((R0,D))
x1 = hcl.placeholder((R1,D))
x_sub_u = hcl.placeholder((R0+R1,D))

with hcl.stage() as s:
	#calclute the mean value of x1 and x2
	k0 = hcl.reduce_axis(0, R0, "k0")
	k1 = hcl.reduce_axis(0, R1, "k1")
	sum0 = hcl.compute((1,D), lambda _k0,y: hcl.sum(x0[k0, y], axis = k0), "sum0")
	sum1 = hcl.compute((1,D), lambda _k1,y: hcl.sum(x1[k1, y], axis = k1), "sum0")
	u0 = hcl.compute((1,D), lambda x,y: sum0[x][y]/R0)
	u1 = hcl.compute((1,D), lambda x,y: sum1[x][y]/R1)
	x0_sub_u0 = hcl.compute((R0,D), lambda x,y: x0[x][y] - u0[0][y])
	x1_sub_u1 = hcl.compute((R1,D), lambda x,y: x1[x][y] - u1[0][y])
	with hcl.for_(0,R0) as i:
		with hcl.for_(0,D) as j:
			x_sub_u[i][j] = x0_sub_u0[i][j]
	with hcl.for_(0,R1) as p:
		with hcl.for_(0,D) as q:
			x_sub_u[R0+p][q] = x1_sub_u1[p][q]
	
	x_sub_u_t = hcl.compute((D,R0+R1), lambda x,y:x_sub_u[y][x])
	k = hcl.reduce_axis(0, D, "k")
	#calculate the covariance matrix sigma using the formula
	#calculate matrix multiplication using hcl.compute
	sigma0 = hcl.compute((R0+R1,R0+R1), lambda x,y: hcl.sum(x_sub_u[x,k]*x_sub_u_t[k,y], axis = k))
	sigma = hcl.compute((R0+R1,R0+R1), lambda x,y:sigma0[x,y]/(R0+R1))
	phi = hcl.compute((1,),lambda x:(1.0/(R0+R1))* R1)
	
o = hcl.create_schedule(s)
print hcl.lower(o, [x0,x1,x_sub_u,phi,sigma,u0,u1])
f = hcl.build(o, [x0,x1,x_sub_u,phi,sigma,u0,u1])
#generate the x0 and x1 points. They must obey Gaussian distribution
mean0 = np.random.randint(low = 0, high = 20, size = D)
cov0 = np.eye(D)
x00 = np.random.multivariate_normal(mean0,cov0,R0)
mean1 = np.random.randint(low = 0, high = 20, size = D)
cov1 = np.eye(D)
x10 = np.random.multivariate_normal(mean1,cov1,R1)
x_sub_u_0 = np.zeros(x_sub_u.shape)

hcl_x0 = hcl.asarray(x00, dtype = hcl.Float())
hcl_x1 = hcl.asarray(x10, dtype = hcl.Float())
hcl_x_sub_u = hcl.asarray(x_sub_u_0, dtype = hcl.Float())
hcl_phi = hcl.asarray(np.zeros(phi.shape), hcl.Float())
hcl_sigma = hcl.asarray(np.zeros(sigma.shape), hcl.Float())
hcl_u0 = hcl.asarray(np.zeros(u0.shape), hcl.Float())
hcl_u1 = hcl.asarray(np.zeros(u1.shape), hcl.Float())

f(hcl_x0, hcl_x1, hcl_x_sub_u, hcl_phi, hcl_sigma, hcl_u0, hcl_u1)
print "--------------hcl_x0------------"
print hcl_x0
print "--------------hcl_x1------------"
print hcl_x1
print "--------------hcl_phi------------"
print hcl_phi
print "--------------hcl_sigma------------"
print hcl_sigma
print "--------------hcl_u0------------"
print hcl_u0
print "--------------hcl_u1------------"
print hcl_u1

