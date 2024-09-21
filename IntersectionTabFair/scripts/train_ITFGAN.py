#!/usr/bin/env python
# coding: utf-8

import sys
import os

# Add the root project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main function from the main.py file
from main import main

if __name__ == "__main__":
    main()
