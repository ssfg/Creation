import base64
import numpy as np

t = np.arange(27, dtype=np.float32)
print "t", t
print t.shape
s = base64.b64encode(t)
print "s", s
r = base64.decodestring(s)
print "r", r
q = np.frombuffer(r, dtype=np.float32)
print "q", q
# m and n are dimensions of original array - use for multidimensional processing
# q = np.reshape(q,(m,n))


print(np.allclose(q, t))
# True

