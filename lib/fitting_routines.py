import math
import numpy as np

## peak fitting routines -- these calculate the delta for each method
def quadraticDelta(a1, a2, a3):
    return (a3 - a1) / (2 * (2 * a2 - a1 - a3))

def barycentricDelta(a1, a2, a3):
    return (a3 - a1)/(a1 + a2 + a3)

def candanDelta(y1, y2, y3, N):
    cor = np.pi/N
    return np.tan(cor)/cor * jacobsenDelta(y1, y2,y3)

def jacobsenDelta(y1, y2, y3):
    return ((y1-y3)/(2*y2-y1-y3)).real

def macLeodDelta(y1, y2, y3):
    R1 = (y1 * y2.conjugate()).real
    R2 = abs(y2)**2
    R3 = (y3 * y2.conjugate()).real
    
    gamma = (R1-R3)/(2*R2+R1+R3);
    return (math.sqrt(1+8*gamma**2)-1)/(4*gamma)

def jainsDelta(a1, a2, a3):
    if (a1 > a3):
        a = a2/a1
        d = a/(1+a)-1
    else:
        a = a3/a2
        d = a/(1+a)
    return d

def quinnsDelta(y1, y2, y3):
    def tau(x):
        return  0.25 * np.log10(3 * x ** 2 + 6 * x + 1) - np.sqrt(6) / 24 * np.log10((x + 1 - np.sqrt(2 / 3)) / (x + 1 + np.sqrt(2 / 3)))
            
    y2r = y2.real; y2i = y2.imag; y2m =abs(y2)**2;

    ap = (y3.real * y2r + y3.imag * y2i) / y2m;
    dp = -ap/(1-ap);
    am = (y1.real * y2r + y1.imag * y2i) / y2m;
    dm = am/(1-am);
    d = (dp+dm)/2+tau(dp*dp)-tau(dm*dm);
    return d

def formalMethodName(method = None):
    if (type(method) != str):
        return "method must be string"
    
    match method.lower():
        case "maximumpixel":
            return "maximum pixel"
        case "quadratic": # weighted average
            return "Quadratic approximation"
        case "barycentric": # weighted average
            return "Barycentric approximation"
        case "jains":
            return "Jain's method"
        case "jacobsen":
            return "Jacobsen's method"
        case "jacobsenmod":
            return "modified Jacobsen's"
        case "macleod":
            return "MacLeod's method"    
        case "candan":
            return "Candan"
        case "quinns2nd":
            return r"Quinn's $2^\mathrm{nd}$ estimator"
        case "gaussian":
            return "Gaussian fit"
        case _:
            return 'unknown method {method}'
