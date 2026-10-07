"""Brine shrimp analysis (Physics M180G, Fall 2026).

The shrimp are tracked in Tracker (physlets.org/tracker) and with SAM 2 / EdgeTAM (segment.py); this
package analyzes the exported tracks.

Modules and who owns them:
    video.py        provided by the course (tested; do not change): read and check phone videos
    convert.py      provided by the course (tested; do not change): make a copy Tracker can open
    segment.py      provided by the course (tested; do not change): track shrimp with SAM 2 or EdgeTAM
    _edgetam.py     provided by the course (tested; do not change): loads EdgeTAM for segment.py
    formats.py      column names and conventions shared by everyone
    load.py         Role A: Tracker files -> one clean table of tracks
    kinematics.py   Role B: speeds, smoothing, jumps (and the tracking noise)
    motion.py       Role B: vx, vy, ax, ay with Tracker's own formulas
    strokes.py      Role C: stroke frequency, per-track summary (and the Reynolds number)
    stats.py        Role C: one value per shrimp, mean and SD, Welch's t-test
    validate.py     Role D: comparing two trackings of the same shrimp
    compare.py      Role D: three tools on one shrimp; two ages over many shrimp (command line provided)
    synthetic.py    not needed this week: fake tracks with known physics
    report.py       not needed this week: the old one-video report
"""

__version__ = "0.3.0"
