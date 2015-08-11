import base64
import numpy as np

t = np.arange(25, dtype=np.float64)
s = base64.b64encode(t)
r = base64.decodestring(s)
q = np.frombuffer(r, dtype=np.float64)

print(np.allclose(q, t))
# True