#!/bin/bash

sphinx-apidoc -f -o rst/ ..
sphinx-build -b html rst/ html/

