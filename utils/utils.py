import re

def transf_duration(x):
    x = re.findall('\d+',x)
    if(len(x)>=2):
        x=int(x[0])*60+int(x[1])
        return x
    else:
        x = int(x[0])*60
        return x