## imports
#import numpy as np
#import matplotlib

### Functions
##
def parse_input (file, field_sep: str = ",", comment_escape: str = "#") -> None:
    "reads a file, splits it into segments using a given separator, removes comments, and forward the result to main"
    import csv
    
    ## reading data
    data = list(csv.reader (file, delimiter = field_sep)) # Crucially list(..)
    
    ## discard comment lines that start with #
    data = [ F for F in data if len(F) > 0 and not F[0][0] == comment_escape ]
    
    ## remove in-line comments
    data_renewed = [ ]
    for F in data:
        G = []
        for f in F:
            pos = f.find (comment_escape)
            if pos > 0:
                G.append(f[:pos])
                continue
            else:
                G.append(f)
        ##
        data_renewed.append(G)
    ##
    return data_renewed

### end of file
