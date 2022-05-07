
# Base equations (recursive)

m = 1.0 / (1.0 + g)

xr = x - (y * r)

y0 = m*(g*xr + s[0])
y1 = m*(g*y0 + s[1])
y2 = m*(g*y1 + s[2])
y3 = m*(g*y2 + s[3])

# Just for reference, here are the state vars:
s[0] = 2.0*y0 - s[0]
s[1] = 2.0*y1 - s[1]
s[2] = 2.0*y2 - s[2]
s[3] = 2.0*y3 - s[3]

# Put them together

y = m*(g*m*(g*m*(g*m*(g*xr + s[0]) + s[1]) + s[2]) + s[3])

y = m*g*(m*g*(m*g*(m*g*xr + m*s[0]) + m*s[1]) + m*s[2]) + m*s[3]
y = (m*m*m*m)*(g*g*g*g)*xr + (m*m*m*m)*(g*g*g)*s[0] + (m*m*m)*(g*g)*s[1] + (m*m)*g*s[2] + m*s[3]
y = m4*g4*xr + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]

y = m4*g4*xr + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]

# Substitute (xr = x - y*r) to remove recursion

y = m4*g4*(x - y*r) + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]
y = m4*g4*x - m4*g4*y*r + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]
y + m4*g4*y*r = m4*g4*x + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]

# Final:
y = ( m4*g4*x + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3] ) / ( 1.0 + r*m4*g4 )

# Factored for multiply-accumulate:
y = ( mg*(mg*(mg*(mg*x + m*s[0]) + m*s[1]) + m*s[2]) + m*s[3] ) / ( 1.0 + r*m4*g4 )

# For an efficient analog-ish style, could use 1992 Rossum paper style saturation - something along the lines of:
y = tanh( mg*(mg*tanh(mg*tanh(mg*tanh(x) + m*s[0]) + m*s[1]) + m*s[2]) + m*s[3] ) / ( 1.0 + r*m4*g4 )


# CEM3328-style resonance compensation:

# Instead of xr = x - (y * r)
# xrg is x resonacnce gain
# I think it's 1.8 on CEM3328, but I'm not sure (see datasheet - this assumes "limiter" block has gain of 1)
xr = x + r*(xrg*x - y)
xr = x + (r  * xrg * x) - (y * r)
xr = x*(1.0 + r*xrg) - (y * r)

# So this is actually the same as 303-style compensation, except on input instead of output

# With nonlinearity, it's
xr = x + r*tanh(xrg*x - y)



