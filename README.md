# Audio Experiments

This is a collection of my "just for curiosity's sake" audio experiments.

## Disclaimer

This is stuff I work on very sporadically and just for fun.
As such, it's not "complete", and never will be - that's not the point.
I don't update it regularly, only when I feel like working on it (especially these days - most of my for-fun audio
coding right now is in C++).

Not everything fully works in all edge cases; test coverage is spotty, and some of existing the tests fail; and there
are quite a few TODOs and FIXMEs.
Code quality & style varies, since this started out as a random collection of old files over the previous 5+ years that
I figured I should consolidate, much of it originally Python 2.
But I figured it's better to put this out there than just letting it languish on my hard drive forever.

For code that's actually meant to be usable, well tested, etc, I'm working on a C++ audio library, some of which is
fairly directly ported from this.
As of writing this, I haven't made any of that code public yet, but I hope to soon (for some value of "soon" :wink:).

## How to use

Various modules here have `main()` functions.
For the sake of dealing with relative imports, all modules get run from `main.py`, with the module name as subcommand -
this will call the module's `main()`.

Many modules also have separate `test()` and/or `plot()` functions. You can run these by adding `--test` or `--plot`.
Some modules just have main call plot and/or test, but others have unique main functionality.

Run with `--help` for a list of arguments that `main.py` takes

### Examples

Run Dattorro/Griesinger "figure 8" reverb algorithm and analyze the impulse response for density, RT60 etc:

    main.py delay_reverb.dattorro_reverb

Process & plot frequency response of linear "4 cascaded poles with feedback" lowpass filter architecture:

    main.py filters.cascade

Analyze various soft & hard clipping functions for harmonic distortion:

    main.py overdrive.overdrive

Iteratively calculate tanh function with feedback, with both finite and infinite open-loop gain:

    main.py overdrive.tanh_fb --plot

Run & analyze various simple compressor algorithm curves:

    main.py compression.basic_compressors

## Performance

As the goal of this code is experimentation, most of it has been written with ease of understanding what's actually
going on over performance.

Many of the classes here are written with a "real-time style" API, i.e. you can feed them audio a bit at a time instead
of needing to process the whole thing at once.
However, the idea behind this was more for the sake of making this easy to port to C/C++ than for actual real-time use.

## FAQ

### Some of your tests fail!

Yeah, I know. I don't really plan on fixing them - again, the point of this is experimentation, not necessarily code
that works in 100% of corner cases.

### Why not just use pytest for your tests?

Yeah, if I was going to start over from scratch, then I'd use pytest.
But I didn't when I started a few years ago, and now I have a basic test framework that works and does the (pretty
simple) things I need it to.

### What about Pedalboard?

Yeah, some of this would probably integrate pretty nicely with
[Spotify's Pedalboard library](https://github.com/spotify/pedalboard/).
But most of this code predates Pedalboard, and I'm not working on this enough anymore that it's really worth adding
Pedalboard integration.

### Why not replace large portions of this with a few lines of numpy/scipy?

Where would the fun be in that?
