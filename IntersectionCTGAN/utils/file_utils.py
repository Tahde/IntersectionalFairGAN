#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/file_utils.py
def print2file(buf, outFile):
    """Write a string buffer to a file."""
    with open(outFile, 'a') as f:
        f.write(buf + '\n')
    print(f"Written to {outFile}: {buf}")


