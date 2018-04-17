from torchtools.meters import AverageMeter


class LossMeter(AverageMeter):
    def on_forward_end(self, trainer, state):
        if state['mode'] != self.mode:
            return
        if self.name in state:
            self.add(state[self.name].data[0])
