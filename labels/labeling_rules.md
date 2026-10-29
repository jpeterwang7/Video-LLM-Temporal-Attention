# How to label attention in DyBench videos

These are the instructions `openai_label.py` sends to the OpenAI API, ahead of hand-labelled examples. Edit
them to change how the API labels; the examples still carry the most weight.

---

You are labelling which parts of a video a person needs to see to answer a multiple-choice question about it.
Label exactly the way the hand-labelled examples are labelled: the same choice of time steps, the same number of
boxes, the same kind of object boxed, and boxes just as tight.

The video is shown as time steps. Each step is one image with two consecutive frames side by side: the left
half is the first frame of the step and the right half the second. Box coordinates are fractions of the LEFT
frame only: x from 0 (its left edge) to 1 (its right edge), y from 0 (top) to 1 (bottom). Steps are numbered
from 1.

1. Read the question, the options and the correct answer, then look at every step.
2. Pick the time steps a person needs to see to answer. In the examples this is almost always every step: all
   of them for moving direction, and all or most of them for reversible dynamics. Leave a step out only when it
   clearly adds nothing, as the examples occasionally do.
3. For each step you pick, draw one box around the object that carries the answer: the thing that moves (moving
   direction) or whose state changes (reversible dynamics). The examples use exactly one box per step.
4. If the object moves between the two frames of a step, make the box cover both of its positions.
5. Keep the box tight around the object, never around background. Include whatever causes the change only when
   the answer depends on it, and hands only when the answer depends on them.
6. If the given answer looks wrong, still label what is needed, and start the note with "WRONG ANSWER:".
7. If you cannot tell what is needed, set unsure to true and say why in the note.
8. Label each video on its own, even if it looks like a reversed version of another video.
