__author__ = 'm'

import os
import os.path
import numpy as np
from ptsa.data.readers import BaseEventReader

from RamPipeline import *


class ControlEventPreparation(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def run(self):
        events = None
        if self.params.include_fr1:
            try:
                e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_FR1', self.pipeline.subject + '_math.mat')
                e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
                events = e_reader.read()
                print "Got FR1 events"
            except IOError:
                pass

        if self.params.include_catfr1:
            try:
                e_path = os.path.join(self.pipeline.mount_point , 'data/events/RAM_CatFR1', self.pipeline.subject + '_math.mat')
                e_reader = BaseEventReader(filename=e_path, eliminate_events_with_no_eeg=True)
                catfr1_events = e_reader.read()
                print "Got CatFR1 events"
                if events is None:
                    events = catfr1_events
                else:
                    print "Joining FR1 and CatFR1"
                    catfr1_events.session = -catfr1_events.session-1
                    fields = list(set(events.dtype.names).intersection(catfr1_events.dtype.names))
                    events = np.hstack((events[fields],catfr1_events[fields])).view(np.recarray)
            except IOError:
                pass

        events = events[events.type == 'PROB']

        print len(events), 'PROB events'

        self.pass_object('control_events', events)
