from RamPipeline import *

import os
import os.path

import numpy as np

from ptsa.data.readers import BaseEventReader


class MathEventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)

    def run(self):
        try:
            e_path = os.path.join(self.pipeline.mount_point, 'data/events', self.pipeline.task, self.pipeline.subject+'_math.mat')
            e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=False)

            events = e_reader.read()
            ev_order = np.argsort(events, order=('session','mstime'))
            events = events[ev_order]

            events = events[events.type == 'PROB']

            self.pass_object(self.pipeline.task+'_math_events', events)

        except IOError:

            self.pass_object(self.pipeline.task+'_math_events', None)

