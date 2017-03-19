# Least Squares Generative Adversarial Networks (LSGAN)
Simple introductory chainer implementation of LSGAN [1].  
If you are using python2, add the extra line on top of the code.
```python
from __future__ import division, print_function
```  

# Training
If you want to run using gpgpu, add `-g 0`.  
```bash
python lsgan.py -g 0
```

# Result
![result](anime.gif)

# Reference
[1] https://arxiv.org/abs/1611.04076
