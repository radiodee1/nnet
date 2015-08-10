import numpy as np

def store_w(w):
    return store(w)
    
    
def store_b(b):
    return store(b)
    
    
def unstore_w(s):
    w = 0
    return w
    
    
def unstore_b(s):
    b = 0
    return b
    
def store(x):
    shape = x.shape
    print 'shape', shape
    ln = len(shape)
    out = []
    
    print(type(x))
    for i in x:
        if  isinstance(i, (list, np.ndarray, tuple)):
            for j in i:
                print 'j', j
                if  isinstance(j, (list, np.ndarray, tuple)) :
                    for k in j:
                        if  isinstance(k, (list, np.ndarray, tuple)):
                            for m in k:
                                out.append(m)
                                print 'm',m
                        else: out.append(k)
                else: out.append(j)
        else :out.append(i)
                
    return out, x.shape
    
    
if __name__ == '__main__':
    x = np.zeros(shape=(4,3,2,2))
    z = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    print x
    print(store_w(x))
