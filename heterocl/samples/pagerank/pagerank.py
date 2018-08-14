"""
This is the pagerank algorithm written in Heterocl.


More information could be found here: https://en.wikipedia.org/wiki/PageRank

=================================
"""
#encoding:utf-8
import numpy as np
import heterocl as hcl

hcl.config.init_dtype = "float32"

#------------------------
#Here we use the website matrix of 39*39 as a sample. Loop for 10000 times.
#The damp value is usually 0.85
N = 39
Loop = 10000
damp = 0.85

A = hcl.placeholder((N,N))
B = hcl.placeholder((N,1))
C = hcl.placeholder((N,1))

with hcl.stage() as s:
	with hcl.for_(0,Loop):
		with hcl.for_(0,N) as p:
			C[p][0] = 0
		with hcl.for_(0,N) as i:
			with hcl.for_(0,N) as j:
				C[i][0] = C[i][0] + A[i][j] * B[j][0]
		with hcl.for_(0,N) as m:
			B[m][0] = C[m][0]
			
o = hcl.create_schedule(s)
print hcl.lower(o, [A, B, C])
f = hcl.build(o, [A, B, C])

#-----------------------------------------------
#Here we generate the website matrix 
a = np.random.rand(N,N)
b = np.linalg.norm(a, ord = 1,axis=0,keepdims=True)
c = a/b
A0 = damp*c + (1-damp)/N*np.ones((N,N),dtype=np.float32)
v = np.random.rand(N,1)
B0 = v/np.linalg.norm(v,1)
B00 = B0
C0 = np.zeros((N,1), dtype=np.float32)

#The following code in annotation could be used to test the accuracy compared to the result of numpy.


hcl_A = hcl.asarray(A0, dtype = hcl.Float())
hcl_B = hcl.asarray(B0, dtype = hcl.Float())
hcl_C = hcl.asarray(C0, dtype = hcl.Float())
f(hcl_A, hcl_B, hcl_C)

# This is the same algorithm in Python using numpy.dot
for i in range(Loop):
	D0 = np.dot(A0,B0)
	B0 = D0


print("--------This is the result of heterocl---------------")
print("The website matrix: ")
print hcl_A
print("The initial vector: ")
print B00
print("The final vector: ")
print hcl_C

print("--------This is the result of Python---------------")
print("The website matrix: ")
print A0
print("The initial vector: ")
print B00
print("The final vector: ")
print D0


