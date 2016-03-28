#!/bin/bash
# check for missing 'raise' before err.Fatal
grep -Rn '^[ ]*err\.Fatal' *
grep -Rn '^[ ]*Exception' *

#check for function signature using default list initialization
grep -Rn '.=\[\]' *
