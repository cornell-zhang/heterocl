import heterocl as hcl
import torch

hcl.init()

def insertionsort_dsl(arr):
    with hcl.Stage("S"):
        with hcl.for_(1, arr.shape[0], name="i") as i:
            key = hcl.scalar(arr[i], "key")
            j = hcl.scalar(i-1, "j")
            with hcl.while_(hcl.and_(j >= 0, key < arr[j])):
                arr[j+1] = arr[j]
                j.v -= 1
            arr[j+1] = key.v

#@torch.jit.script
def insertionsort_python(arr):
    length = arr.shape[0]
    for i in range(1, length):
        key = arr[i]
        j = i - 1
        while j >=0 and key < arr[j]:
           arr[j + 1] = arr[j]
           j -= 1
        arr[j + 1] = key

A = hcl.placeholder((10,), "A")
s1 = hcl.create_schedule([A], insertionsort_python)
print(insertionsort_python.code)

#s2 = hcl.create_schedule([A], insertionsort_python)

print(hcl.lower(s1))
#print(hcl.lower(s2))

