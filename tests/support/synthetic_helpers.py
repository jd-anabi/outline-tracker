"""A scene with one object, for the tests of the synthetic clips.

`one_object_scene(shape, path)` is a small scene of one object A on the default background: the ground truth tests
of tests/synthetic/test_synthetic.py and the tests of the encoded clips in tests/synthetic/test_synthetic_clips.py
both build their scenes with it. Sizes are in px, frames are video frame numbers.
"""

from outline_tracker.synthetic.clips import Scene, SceneObject


def one_object_scene(shape, path, size=(120, 90), n_frames=4, **more):
    return Scene(size=size, fps=240.0, n_frames=n_frames, objects=(SceneObject("A", shape, path),),
                 mm_per_px=0.01, **more)
