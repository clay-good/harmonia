# Why input variability is the load-bearing idea

The safety call on a drug's heart risk can flip from "low" to "high" without
anyone changing the model, the math, or the patient. All it takes is believing a
different lab's number for the same measurement. Harmonia is built to make that
fact impossible to ignore.

## The short version

In-silico proarrhythmia assessment has a clean story and a dirty secret.

The clean story: measure how strongly a drug blocks a handful of cardiac ion
channels, feed those numbers into a published heart-cell model, simulate the
heartbeat, read off a risk score. The FDA-backed CiPA initiative built exactly
this pipeline, and it works.

The dirty secret: the numbers you feed in are not stable. The same drug, the same
channel, measured in two different labs, routinely gives IC50 values that differ
several-fold. Sometimes the drug barely blocks the channel at all, and the "IC50"
quoted in the paper is an extrapolation of an extrapolation. Garbage in is not
quite garbage out here, because the model is good. It is more like fog in, verdict
out. The model faithfully propagates your uncertainty into the answer, and then
everyone reads the answer as if it were certain.

Harmonia's whole reason to exist is to carry that fog all the way to the end and
show it to you.

## What an IC50 actually is, and why it lies

IC50 is the concentration of a drug that blocks half of a channel's current. Lower
IC50 means a more potent blocker. For the channel that matters most in cardiac
safety, hERG (which carries the repolarizing potassium current IKr), a low IC50 is
a red flag.

To measure it, you apply increasing drug concentrations and watch the current
shrink. Then you fit a curve (the Hill equation) and read off the half-block
point.

Here is the trap. To locate the halfway point of a curve, you need to see enough
of the curve. If the highest concentration you tested only knocked the current
down by 40%, you never saw half-block. You are fitting a curve to its own opening
notes and guessing where the chorus lands. The fitted IC50 in that case is not a
measurement. It is a hope with error bars wide enough to drive a truck through.

The field has a rough rule: if the maximum block you observed is below about 60%,
the IC50 is not identifiable. Yet these unidentifiable values still appear in
tables as single numbers, stripped of the warning, and get fed into models as if
they were real.

Harmonia refuses to do that. Any record whose maximum observed block is under 60%
is forced to Tier D, tagged unidentifiable, excluded from the simulation, and made
to cap the trust level of every assessment that touches it. The worked example in
the dataset is ranolazine's block of the L-type calcium channel: 35% maximum
block, so the IC50 of 296,000 nM is kept only for provenance and never used to
classify anything. The honest answer to "what is ranolazine's calcium IC50" is
"we do not know," and Harmonia says so in a field a machine can read.

## Variability is not noise to be averaged away

The instinct, when three labs report 4.9, 6.6, and 4.0 nM for the same drug, is to
average them into one tidy number and move on. That instinct destroys the most
important information in the dataset.

The spread is not measurement noise around a true value. It is a real, structural
disagreement driven by how the measurement was made: manual patch clamp versus an
automated chip, room temperature versus body temperature, one cell line versus
another. A several-fold spread in IC50 can be the difference between a drug that
prolongs the heartbeat dangerously and one that does not.

So Harmonia keeps every source value, with its platform and its citation, as a
first-class field. It computes the fold-range and the interquartile range and
stores them. When you ask for a risk assessment, it does not pick a number. It
samples across the whole spread, runs the simulation for each draw, and hands you
back a distribution of outcomes plus one blunt statistic: how often the
high/intermediate/low classification flipped.

That flip frequency is the product. A drug with a 10% flip frequency is a drug
whose risk call you can mostly trust. A drug with a 55% flip frequency is a coin
toss dressed up as a conclusion, and you deserve to know that before you act on it.

## The picture that makes it concrete

Take two drugs through the same kernel.

Dofetilide is almost a pure hERG blocker with a tight, well-agreed IC50. Its
distribution of action-potential prolongation sits squarely in the high-risk zone
and barely moves when you resample the inputs. Flip frequency around 10%. The
model is confident because the inputs are.

Verapamil is the interesting one. It blocks hERG too, which should make it
dangerous, but it also blocks the L-type calcium channel, which shortens the
action potential and pulls risk back down. The two effects nearly cancel, which
leaves verapamil sitting right on the boundary between low and intermediate. Now
add the inter-lab spread in its IC50 values, and the classification falls on
whichever side the dice land. Flip frequency above 50%.

Verapamil is genuinely low-risk in the clinic. The point is not that the kernel
nails the label. The point is that the kernel, honestly run, tells you the label
is unstable, which is the truth, and which a single confident number would have
hidden.

## The honest limits

Harmonia's bundled heart-cell model is a reduced one. It has the right currents
and the right qualitative behavior (block hERG and the heartbeat lengthens; block
calcium and it shortens again), but it is not the bit-exact, regulatory-grade
O'Hara-Rudy model. So it ships at Tier C.

Its default risk metric is qNet, the integrated-charge measure CiPA prefers, and
getting qNet to work took a small piece of real physiology. In a model with no
ion pumps, the charge over a full paced beat is conserved, which makes the qNet
sum stubbornly insensitive to the very currents you care about. Adding a Na-Ca
exchanger and then excluding it from the qNet sum breaks that conservation, and
qNet starts discriminating: lower qNet, higher risk, exactly as the field expects.

On the twelve CiPA training drugs the reduced qNet classifier recovers ten labels.
More telling, across all twenty-eight CiPA compounds it never makes a two-category
error. It never calls a high-risk drug low or a low-risk drug high. On a safety
screen that is the property that matters. The exact three-way accuracy on the
sixteen-drug validation set is honestly worse, because many of those drugs have a
free plasma concentration so low that four times it still barely touches the
channel, and no metric reads much signal from almost no block. The classic
action-potential-prolongation metric is still there, one argument away
(metric="apd90"), and it does worse than qNet on both counts.

None of that weakens the thesis. The classifier is a demonstrator. The machinery
that carries input variability through to a flip frequency is the contribution,
and it is correct no matter how good or bad the underlying classifier is. Swap in
the full ORd model in a later phase and the same plumbing produces a sharper
answer. The fog-tracking does not change. Only the resolution does.

## Why this is worth building

The uncertainty-quantification papers in this field already said all of this. They
showed, carefully, that input variability dominates the proarrhythmia answer. What
they did not do, what nobody did, was ship it as curated infrastructure: a dataset
where the spread is a field, the unidentifiable values are flagged, the worst input
caps the trust, and the whole thing exports into the standard cardiac-modeling
formats with a permanent "not a clinical verdict" stamp on every file.

A model is only as trustworthy as its weakest, least-validated input. Most tools
hide that input. Harmonia makes it the first thing you see.
