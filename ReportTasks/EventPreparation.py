from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
from ptsa.data.readers import BaseEventReader
import os
import numpy as np


class EventPreparation(ReportRamTask):
    def __init__(self,tasks,mark_as_completed=False,name=None):
        super(ReportRamTask,self).__init__(mark_as_completed,name=name)
        self.tasks=tasks
        self.events = None

    def load_events(self,task):
        '''
        I was going to name this load_task_events, but then I realized that there are "task_events", and that
        is not strictly what this method is loading.
        :param task: The RAM task whose events we are loading
        :return: events
        '''
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp)==1 else int(tmp[1])
        mount_point = self.pipeline.mount_point

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        event_files = sorted(list(json_reader.aggregate_values('all_events', subject=subj_code, montage=montage, experiment=task)))
        events = [BaseEventReader(filename=os.path.join(mount_point,event_file)).read() for event_file in event_files]
        return np.concatenate(events,0).view(np.recarray)

    def pass_events(self):
        events=self.events
        self.pass_object('all_events',events)
        self.pass_object('events', events[events.type=='WORD'])

    def run(self):
        self.events = np.concatenate([self.load_events(task) for task in self.tasks],0).view(np.recarray)
        self.pass_events()



