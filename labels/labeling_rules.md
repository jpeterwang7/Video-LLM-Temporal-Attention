# How to label attention in DyBench videos

These are the instructions `openai_label.py` sends to the OpenAI API, ahead of hand-labelled examples. Edit
them to change how the API labels; the examples still carry the most weight.

---

You are labelling which parts of a video a person needs to see to answer a multiple-choice question about it.
Label exactly the way the hand-labelled examples are labelled: the same choice of time steps, the same number of
boxes, the same kind of object boxed, and boxes just as tight.

The video is shown as time steps. Each step is one image with two consecutive frames side by side: the left
frame is the first of the step and the right frame the second. Box coordinates are fractions of the LEFT frame
only: x from 0 (its left edge) to 1 (its right edge), y from 0 (top) to 1 (bottom). The white margins along the
top and left of the left frame are a scale marked in those fractions, with a tick every 0.1: read each box's
coordinates off it. Steps are numbered from 1.

1. Read the question, the options and the correct answer, then look at every step.
2. Pick every time step. Leave one out only when it clearly adds nothing, which is rare: the examples pick all
   steps for moving direction and nearly always for reversible dynamics. If unsure about a step, pick it.
3. For each step, draw exactly one box, around the object that carries the answer: the thing that moves or is
   asked about (moving direction), or whose state changes (reversible dynamics).
4. Place each box carefully, frame by frame. Most errors in earlier labels were boxes on the right object but in
   the wrong place: beside it, or above it. Find the object in that step's left frame, read its left, right, top
   and bottom edges off the scale, and check that the box's centre is on the object and that the whole object
   is inside. Do not reuse a position from another step unless the object really has not moved.
5. Synthetic scenes (plain grey floor with cubes, spheres and cylinders) need the most care. Objects are small,
   often a few percent of the frame, and a small error misses them completely. Identify the object by the colour
   and shape in the question, not by position, and box it even when it sits at the edge of the frame or does not
   move.
6. If the object moves between the two frames of a step, make the box cover both of its positions.
7. Keep the box tight around the object, with little background. For a person or animal, box that one body, not
   the area around it. Include whatever causes the change only when the answer depends on it, and hands only when
   the answer depends on them.
8. If the given answer looks wrong, still label what is needed, and start the note with "WRONG ANSWER:".
9. If you cannot tell what is needed, set unsure to true and say why in the note.
10. Label each video on its own, even if it looks like a reversed version of another video.
