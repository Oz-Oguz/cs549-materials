# Demonstrations

Start `python3 launch.py` from this extracted folder to use the browser apps.

[Return to the companion index](README.html).

The Panda commands below need the [Drake environment](README.html#drake). Each web page opens without running anything: a replay waits for Step, and the IK page solves only when the target changes. Open the printed address; stop with Ctrl+C.

## Prediction versus FK on the Panda {#gn-ik-replay}

```bash
python launch.py gn-ik-replay
```

## Hard-target IK and its solver results {#ik-nlp}

```bash
python launch.py ik-nlp
```

## The Panda elbow limit and multiplier {#constrained-ik-replay}

```bash
python launch.py constrained-ik-replay
```

## Hard-target rail and joint-bound multiplier events {#ik-nlp-al-replay}

```bash
python launch.py ik-nlp-al-replay
```

## Browser app starting states

- [modules/gauss-newton-ik/index.html?view=local-step](modules/gauss-newton-ik/index.html?view=local-step)
- [modules/gauss-newton-ik/index.html?y=0.5,0.2&method=gd](modules/gauss-newton-ik/index.html?y=0.5,0.2&method=gd)
- [modules/gauss-newton-ik/index.html?y=0.5,0.2&method=gn](modules/gauss-newton-ik/index.html?y=0.5,0.2&method=gn)
- [modules/kkt-explorer/index.html?mode=disc&target=0.5,0.3](modules/kkt-explorer/index.html?mode=disc&target=0.5,0.3)
- [modules/kkt-explorer/index.html?mode=wall&target=1.6,0.6](modules/kkt-explorer/index.html?mode=wall&target=1.6,0.6)
- [modules/opt-explorer/index.html?tab=concepts&fn=armijo-wolfe&preset=start-x-2&method=gd&ls=strong-wolfe&arrows=off](modules/opt-explorer/index.html?tab=concepts&fn=armijo-wolfe&preset=start-x-2&method=gd&ls=strong-wolfe&arrows=off)
- [modules/opt-explorer/index.html?tab=concepts&fn=elongated&preset=classic-zig-zag&method=gd&ls=armijo&alpha=1&arrows=on](modules/opt-explorer/index.html?tab=concepts&fn=elongated&preset=classic-zig-zag&method=gd&ls=armijo&alpha=1&arrows=on)
- [modules/opt-explorer/index.html?tab=concepts&fn=elongated&preset=classic-zig-zag&method=gd&ls=off&alpha=0.08&arrows=off](modules/opt-explorer/index.html?tab=concepts&fn=elongated&preset=classic-zig-zag&method=gd&ls=off&alpha=0.08&arrows=off)
- [modules/opt-explorer/index.html?tab=concepts&fn=quartic-saddle&preset=near-the-maximum&method=newton&ls=off&arrows=off](modules/opt-explorer/index.html?tab=concepts&fn=quartic-saddle&preset=near-the-maximum&method=newton&ls=off&arrows=off)
- [modules/opt-explorer/index.html?tab=concepts&fn=rosenbrock&preset=classic-start&method=bfgs&ls=strong-wolfe&arrows=off](modules/opt-explorer/index.html?tab=concepts&fn=rosenbrock&preset=classic-start&method=bfgs&ls=strong-wolfe&arrows=off)
- [modules/opt-explorer/index.html?tab=concepts&fn=rosenbrock&preset=classic-start&method=lbfgs&ls=exact&arrows=off](modules/opt-explorer/index.html?tab=concepts&fn=rosenbrock&preset=classic-start&method=lbfgs&ls=exact&arrows=off)
- [modules/opt-explorer/index.html?tab=trajectory&preset=ik](modules/opt-explorer/index.html?tab=trajectory&preset=ik)
- [modules/opt-explorer/index.html?tab=trajectory&preset=ik&elbow=60](modules/opt-explorer/index.html?tab=trajectory&preset=ik&elbow=60)
- [modules/opt-explorer/index.html?tab=trajectory&preset=ik&y=0.7,0](modules/opt-explorer/index.html?tab=trajectory&preset=ik&y=0.7,0)