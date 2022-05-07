
f = 0.5*f

yHp = x - yLp - q*yBp
yBp = f*yHp + s0
yLp = f*yBp + s1

# best:
s0 = f*yHp + yBp
s1 = f*yBp + yLp

# works but bad:
s0 = yBp
s1 = yLp

# Combining terms to get rid of yBp:

yHp = x - yLp - q*(f*yHp + s0)

yLp = f*(f*yHp + s0) + s1
    = f*f*yHp + f*s0 + s1

# Combining terms to get rid of yLp:

yHp = x - (f*f*yHp + f*s0 + s1) - q*(f*yHp + s0)
    = x - f*f*yHp - f*s0 - s1 - q*f*yHp + q*s0
    = - f*f*yHp - q*f*yHp + x - f*s0 - s1 + q*s0

yHp + f*f*yHp + q*f*yHp = x - f*s0 - s1 + q*s0

yHp*(1 + f*f + q*f) = x - f*s0 - s1 + q*s0

yHp = (x - f*s0 - s1 + q*s0) / (1 + f*f + q*f)
    = (x - f*s0 + q*s0 - s1) / (1 + f*f + q*f)
    = (x - (f + q)*s0 - s1) / (1 + f*f + q*f)

# However, according to Will Pirkle, this is the formula:
# Also, his s0 & s1 determination is different
# his:
#   s0 = g*yHp + yBp
#   s1 = g*yBp + yLp

yHp = (x - 2*R*s0 - f*s0 - s1) / (1 + 2*R*f + f*f)

# Differences:
# q -> 2R

yHp = (x - q*s0 - f*s0 - s1) / (1 + q*f + f*f)

# The other difference is 

# Working backwards:

yHp + q*f*yHp + f*f*yHp = x - q*s0 - f*s0 - s1
yHp = x - f*f*yHp - f*s0 - s1 - q*f*yHp - q*s0
yHp = x - (f*f*yHp + f*s0 + s1) - q*(f*yHp + s0)