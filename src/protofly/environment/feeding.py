"""Minimal feeding environment for ProtoFly.

The environment represents whether sugar is available and how much
remains.

It contains no neural dynamics, learning, reward, or biological
interpretation of motor activity.
"""


class FeedingEnvironment:
    def __init__(self, sugar_present=False, sugar_amount=0.0):
        self.sugar_present = bool(sugar_present)
        self.sugar_amount = max(0.0, float(sugar_amount))

        if self.sugar_amount <= 0:
            self.sugar_present = False

    def set_sugar(self, present, amount=None):
        self.sugar_present = bool(present)

        if amount is not None:
            self.sugar_amount = max(0.0, float(amount))

        if self.sugar_amount <= 0:
            self.sugar_present = False

    def sugar_contact(self):
        return self.sugar_present and self.sugar_amount > 0

    def consume_sugar(self, amount):
        """Remove sugar from the environment.

        This is an environmental operation only. Deciding when the fly
        performs this action belongs to the actuator layer.
        """
        amount = max(0.0, float(amount))

        consumed = min(amount, self.sugar_amount)
        self.sugar_amount -= consumed

        if self.sugar_amount <= 0:
            self.sugar_amount = 0.0
            self.sugar_present = False

        return consumed