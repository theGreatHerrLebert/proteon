# Why Proteon Exists

## A Personal Starting Point

Let's be honest for a moment.

I like writing code. Most of the time, I genuinely enjoy it. But what I enjoy far more is seeing the code run. As a scientist, what I really care about is seeing it produce something that actually means something. The
moment where computation turns into insight. Where the system does what you hoped it would, and you
can start interpreting results instead of fighting implementations.

Then something changed.

Within a short period of time, AI-assisted coding went from a curiosity to something I was actively
relying on. Writing code was no longer the bottleneck. At first, it felt incredible. Then it felt
dangerous. And then it failed.

In the span of four weeks, I generated more code than I had written in the previous year.
Everything compiled. Everything looked reasonable. Everything ran. What I ended up with was a system that appeared sound. In reality, it was untested, entangled, and deeply opaque.

The worst part was not the complexity. The worst part was this: I could not take responsibility for
what I had built, because I did not fully understand it.

## Capability Without Control

That experience forced a shift in perspective. Up to that point, I had assumed that writing code was the hard part. That if something compiled, ran, and produced plausible results, I was on the right track. The problem was not that the system failed to run. The problem was that I had no reliable way
of knowing whether it was correct.

The limiting factor is no longer writing code. It is trusting the system you built.

There was also a moment of real discomfort. It felt like I suddenly had access to capabilities that
had previously been out of reach: cleaner architecture, faster runtimes, components I would not have
attempted myself. Individually, these were exactly the things I had always wanted. But taken
together, they exposed a problem I had not anticipated: I could assemble a system that looked
correct and performed well without being able to verify that it actually behaved correctly.

AI-assisted coding changes what it means to build software responsibly. When developers rely
increasingly on generated implementations, the old habit of trusting code because it was manually
authored becomes less defensible. This does not reduce responsibility. It increases it. The burden
shifts from manual authorship to behavioral validation, from implementation familiarity to
reproducible verification. This is especially important in scientific computing, where results can
look plausible long before they are correct.

## Testing and Its Limits

My first reaction was simple: test everything. If generating code is cheap, testing it should be
cheap as well. So I started enforcing it systematically, and with AI assistance, this became almost
frictionless.

But there is a fundamental limitation. Unit tests can tell you that your code is consistent with
your expectations. They cannot guarantee that your expectations are correct. A function can pass every test you wrote and still be wrong, because the tests themselves encode your assumptions.

This is where the notion of an oracle becomes necessary. Instead of only testing code against
itself, you test it against established tools, known implementations, and large real-world datasets.
The question changes from "does this function behave as expected?" to "does this system agree with
something we already trust?" Internal consistency is necessary but not sufficient. Correctness has
to be established relative to something external.

In practice, for proteon, this means numbers like these: the AMBER96 implementation matches
OpenMM to within 0.2% on every energy component at NoCutoff (218/218 invariants). The TM-align
port sits within a 0.003 median TM-score drift of the reference C++ USalign across 4,656 pairs.
The SASA implementation agrees with Biopython to 0.17% median deviation on 1,000 structures.
And the end-to-end CHARMM19+EEF1 pipeline processed a 50,000 random-PDB battle test at 99.1%
success in 3.5 hours on a single RTX 5090. Without external oracles, none of these numbers mean
anything. With them, every regression has somewhere concrete to be caught.

## Why This Is Hard, and What Proteon Does About It

In structural bioinformatics, building a consistently testable system is surprisingly difficult. The
functionality needed to construct pipelines is fragmented: structure parsing in one library,
alignment in another, geometry in a third, preparation somewhere else. Each comes with different
interfaces, assumptions, and formats. Even if all components are individually correct, combining them into a coherent, testable system is non-trivial. And testing the system as a whole against a consistent reference becomes difficult to do systematically.

Proteon's response is to consolidate the parts that are widely used, conceptually stable, and already well understood, and bring them into a single framework where they can be composed easily, executed at scale, and validated against strong external references as a core requirement rather than an afterthought. The goal is not
to recreate functionality. It is to make that functionality reliably usable under rapid iteration.

## On Credit and Invisible Debt

The Acknowledgements section of this project tries to be thorough. The tools that serve as oracles
are named. The papers behind the algorithms are cited. The people whose implementations made this
possible are credited by name. That part is tractable because the sources are known.

AI-assisted development adds a layer that is harder to account for. When a model generates a
solution, it draws on a large amount of human work absorbed during training: code from public
repositories, ideas from blog posts and papers, design patterns that someone worked out carefully
and shared openly. By the time any of it reaches your codebase, the original authors are not
visible. There is no import statement to follow, no citation to add.

The scientific community runs on something less formal than licenses (but of course licenses are
still part of it): a shared understanding that ideas have origins and that people deserve credit
for their contributions. We are a somewhat
strange family of largely unrelated people, held together more by norms than by contracts (that
are non-permanent for most of us anyways). Those norms are worth respecting even when nothing
formally requires it.

The practical response is not complicated. Credit what you can trace, be transparent about how the
work was done, and treat the Acknowledgements as a genuine attempt at accounting rather than a
formality. That is what this project tries to do.

The broader question of how attribution should work under AI-assisted development is one the
scientific community will figure out over time, the same way it worked out norms around data
sharing and reproducibility. It is a real question worth taking seriously, but hardly a reason to
stop building things.

## Trust as a Design Principle

Good software design has always orbited around abstraction. We build layers so we do not have to
think about everything at once. But abstraction only works under one condition: the thing you
abstract over must be trustworthy. If a component is inconsistent, poorly defined, or unvalidated,
it does not function as an abstraction. It becomes a liability.

We already accept this implicitly. We rely on code we did not write. Numerical libraries, plotting tools, parsers. We use them without inspecting every line. We trust them not because of authorship,
but because of evidence: they have been tested extensively, used at scale, and validated over time.

What changes with AI-assisted development is not the need for trust. It is the rate at which
unverified code can enter a system. Complexity can now be generated faster than it can be
validated.

That is the gap this project is concerned with.

Proteon is one attempt to close that gap, by making validation explicit, reproducible, and central to the design.

## On AI-Assisted Development, and Why This Project Exists

Proteon was developed with the help of AI-assisted coding. Some components, like the performance-critical CUDA kernels, would have been significantly more difficult to implement alone. This is not an exception. It reflects how software is increasingly built.

The relevant question is not whether the code was written entirely by hand. It is whether the
resulting system has been validated. In this project, trust is not derived from authorship. It is
derived from validation against established tools, large-scale testing on real data, and consistent
behavior under defined conditions.

Proteon exists for two reasons. First, it is the tool I wanted to have: fast, composable
structural computation with consistent interfaces and validated building blocks, without forcing a
platform or workflow. Second, it is a testbed, a place to explore what responsible software development looks like in a world where code can be generated rapidly, systems grow faster than they can be inspected line by line, and complexity is no longer limited by implementation effort.

The question is no longer whether we use these tools. The question is how we build systems that
remain trustworthy under these conditions.

This project was built with the help of AI-assisted coding. All mistakes are still mine.
