

# Notes on linearity & frequency response



## LTI vs just "linear"

First, a note on terminology:

If a system is non-LTI, does it matter if it's time-variant (but possibly linear) vs nonlinear (but possibly
time-invariant)? Maybe. One example where it does is Volterra series modelling, which works well on nonlinear systems
but fails for time-varying ones.

But for the sake of this frequency analysis, we're not really interested in time-varying systems, so most measurements
here will assume the system is time-invariant (but possibly nonlinear). Of course, time variance may appear as
nonlinearity (and technically, you can't necessarily distinguish between the two - though in practice, you usually can).

## What does frequency response even mean for a nonlinear system?

Technically, frequency response is *only* defined for linear systems.
What does frequency response even mean when a system is going to add components at other frequencies?
Do we want the amplitude of the _total_ output signal vs the input signal, _just of that frequency component_?

Impulse response analysis doesn't really make sense for a nonlinear system, because convolution will not work.

Conceptually, running white noise through the system makes a lot of sense here. The problem is, it's random, so you
won't get a perfect response. You have to average over a pretty long time to get a decent response, and this slows
things down quite a lot. This also doesn't help us calculate THD+N or other similar stats.

The definition that makes the most sense here is to see the system's effect on sine waves at many frequencies. In other
words, run a sine wave through the system and measure the output signal. Unfortunately, this is slow, but it's the best
we can do (and at least it's not as slow as the noise method).

We can measure the following things on the output:
* Magnitude & phase - based on sine & cosine components at that frequency (i.e. DFT)
* RMS - relative to the RMS of the input wave, i.e. (sine amplitude - 3.01 dB)
* THD+N - total harmonic distortion + noise
* THD only - trickier to measure than THD+N, but doable

Should THD be relative to input or output? Output probably makes more sense here, but ultimately you can calculate one
from the other quite easily.

## How can we possibly test linearity?

First of all, we're talking discrete digital signals, so technically all non-trivial systems will be nonlinear due to
precision errors. What we care about is if a system is **reasonably** linear - i.e. the only nonlinearites are due to
numerical error. 

Second, it's impossible to prove any black-box system is linear without being able to run every infinitely many signals
through it. For example, even if we could prove a system could take a sine wave of amplitude 1,000,000 completely
linearly, what if that system clips at 1,000,001? So we have to also add "within a reasonable amplitude" to our
definition of "reasonably linear".

(The equivalent would be true for time variance as well, were we to measure that: we would have to at least run an
infinitely-long signal, so at best we could only prove that it is time-invariant within a certain time range.)

However, amplitude isn't the only element requiring infinitely many signals. For example, imagine a system that is
completely linear except on values between 0.500001 and 0.500002? A test will only catch this if the input signal
happens to have a sample in that range. (This problem wouldn't exist for continuous-time signals thanks to the
intermediate value theorem, but alas, we're in the discrete domain here.)

For that matter, a nonlinearity could even be something completely different - it could be a finite state machine
that behaves completely linearly until it sees 3 specific sample values in a row and then changes behavior.
Obviously we can't test for this without infinitely many signals either.

So we will have to do the best we can -we want to test if the system is _reasonably_ linear, within a specified
amplitude, for the types of signals we expect to see.

So how do we actually achieve this?

### The superposition test

Formally, linearity is defined by 2 criteria, together known as the superposition principle:
* **Homogeneity:** `f(a*x) = a*f(x)`
* **Additivity:** `f(x1 + x2) = f(x1) + f(x2)`

This is actually quite easy to test, and in most cases should work. But if we want to be thorough, it's worth testing
other criteria as well, in order to catch a greater range of possible nonlinearities. Especially since this still may
not catch some very basic nonlinearities:
* It won't catch `y = abs(x)` unless you ensure to test a negative value
* It won't catch hard clipping unless you ensure you test a value greater than the clip threshold
* It won't catch crossover distortion with hard thresholding unless you test a value under the threshold

### Run a sine wave through and test that the output has no THD+N?

There are a few problems with that.

First of all, it's slow. Yeah, we'll use this method to measure frequency response if a system is nonlinear - but how do
we know the system is nonlinear in the first place? Here we want a relatively quick test that we can use to check if
linear before we decide whether we have to do this instead of a much faster method (i.e. impulse response analysis). 

Second, and more importantly, just because it is linear at one frequency does not mean it would be at all frequencies.
So we would have to test many different frequencies of sine waves. For example, if a system is a filter with a high Q
fed into a hard clipper, then the system will probably appear linear everywhere except at frequencies close to the
resonant frequency. Again, even though we do this anyway when we calculate frequency response, we want something we can
quickly check before doing this.

### Test that the derivative of the step response equals the impulse response?

Perhaps surprisingly, this doesn't always work - not even for some "reasonable" systems.

For example, think about a basic overdrive of the form `y = tanh(x)`.

This is a great example of a nonlinear system: **every** derivative is continuous, which means it has an infinite number
of harmonics, and at **all** input signal amplitudes (unlike hard clipping). So it should be really easy to detect as
nonlinear, right? Well, this method doesn't even catch this. Say we run an impulse into it:
   
    impulse =      [ 1.0    , 0.0    , 0.0    , 0.0     ]
    f(impulse) =   [ 0.76159, 0.0    , 0.0    , 0.0     ]

    step =         [ 1.0    , 1.0    , 1.0    , 1.0     ]
    f(step) =      [ 0.76159, 0.76159, 0.76159, 0.76159 ]
    d/dt f(step) = [ 0.76159, 0.0    , 0.0    , 0.0     ] == f(impulse)

In fact, this problem exists for _all_ memoryless systems.

So clearly this method will not work.

### What about the ramp response?

    ramp =           [ 1.0       ,  2.0       ,  3.0       ,  4.0        ]
    f(ramp) =        [ 0.76159416,  0.96402758,  0.99505475,  0.99932930 ]
    d/dt f(ramp) =   [ 0.76159416,  0.20243342,  0.03102717,  0.00427455 ] != f(step)
    d2/dt2 f(ramp) = [ 0.76159416, -0.55916073, -0.17140625, -0.02675263 ] != f(impulse)

This fixes the problem!
You're essentially running 1 of every possible input amplitude in (within a certain range, and reasonable quantization).
That's a great way to catch most nonlinearities.

##### This still isn't perfect...

Unfortunately, this isn't perfect either. What about a slew limiter? This is nonlinear, but whether or not a ramp test
will catch it depends on the ramp's slope. But a step response should catch this easily.

### So... some combination?

Yeah, we pretty much have to.

So we can compare all of the following:

* `d/dt f(step) == f(impulse)`
* `d/dt f(ramp) == f(step)`
* `d2/dt2 f(ramp) == f(impulse)`

From what I can tell, I think this should catch all reasonable nonlinearities. (I mean, sure, you can come up with other
failing cases - what about a 2nd-order slew limiter that clips the 2nd derivative? What about the same but 7th-order?)

Checking all 3 might be redundant - I think I might be able to narrow this down to just 2 checks. After all, if
`ramp' = step` and `step' = impulse`, shouldn't it follow that `ramp'' = impulse`? But since you need to calculate all 3
responses anyway, it's really easy to just check all 3 against each other regardless. Also, we'll need all 3 because of:

##### What about negatives?

What about the function `y = abs(x)`?
So far, none of these tests will catch this, because we've only looked at positive values.
We're looking to test in the range of magnitude <= 1, but so far we've only tested the range [0, 1], not [-1, 1]

Long story short, we can continue to use the same 3 above checks; we just flip the step response in the comparison
