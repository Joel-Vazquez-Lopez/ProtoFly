"""Minimal feeding environment for ProtoFly Experiment 01.

The environment contains no language, learning, or reward.
It only represents whether the fly's sugar-sensing organs are
currently exposed to sugar.
"""


class FeedingEnvironment:
    """Minimal binary sugar-contact environment."""

    def __init__(self, sugar_present=False):
        self.sugar_present = sugar_present

    def set_sugar(self, present):
        """Set whether sugar is currently contacting the fly."""
        self.sugar_present = bool(present)

    def sugar_contact(self):
        """Return whether the sugar sensory pathway should be stimulated."""
        return self.sugar_present