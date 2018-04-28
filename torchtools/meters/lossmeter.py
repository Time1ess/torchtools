from torchtools.meters import AverageMeter


class LossMeter(AverageMeter):
    """Meter that measures average loss for epoch."""
    def on_forward_end(self, trainer, state):
        if state['mode'] != self.mode:
            return
        if self.name in state:
            self.add(state[self.name])
