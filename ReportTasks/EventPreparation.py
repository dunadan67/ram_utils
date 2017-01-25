from ReportUtils import ReportRamTask
from ptsa.data.readers.IndexReader import JsonIndexReader
from ptsa.data.readers import BaseEventReader
import os
import numpy as np


class EventPreparation(ReportRamTask):
    def __init__(self,tasks,mark_as_completed=False,name=None):
        super(ReportRamTask,self).__init__(mark_as_completed)
        self.tasks=tasks
        self.events = None
        self.set_name(name)

    def read_events(self,task):
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

    def load_events(self):
        self.events = np.concatenate([self.read_events(task) for task in self.tasks], 0).view(np.recarray)

    def pass_events(self):
        task = self.pipeline.task
        events=self.events
        self.pass_object('all_events',events)
        self.pass_object(task+'_events', events[events.type=='WORD'])

    def run(self):
        self.load_events()
        self.pass_events()


class FR1EventPreparation(EventPreparation):
    def pass_events(self):
        events=self.events
        math_events = events[events.type == 'PROB']
        rec_events = events[events.type == 'REC_WORD']
        intr_events = rec_events[(rec_events.intrusion!=-999) & (rec_events.intrusion!=0)]

        self.pass_object('math_events', math_events)
        self.pass_object('intr_events', intr_events)
        self.pass_object('rec_events', rec_events)

        super(FR1EventPreparation,self).pass_events()


class JointFR1EventPreparation(FR1EventPreparation):
    def __init__(self):
        super(JointFR1EventPreparation, self).__init__(['FR1', 'catFR1'], False)

    def load_events(self):
        self.fr1_events,self.catfr1_events = (self.read_events(task) for task in self.tasks)
        self.catfr1_events.session+=100
        ev_fields=['item_num', 'serialpos', 'session', 'subject', 'rectime', 'experiment', 'mstime', 'type', 'eegoffset', 'iscorrect', 'answer', 'recalled', 'item_name', 'intrusion', 'montage', 'list', 'eegfile', 'msoffset']
        self.events = np.concatenate([event[ev_fields] for event in (self.fr1_events,self.catfr1_events)]).view(np.recarray)

    def pass_events(self):
        self.pass_object('cat_events',self.catfr1_events[self.catfr1_events.type=='WORD'])
        super(JointFR1EventPreparation, self).pass_events()

class PAL1EventPreparation(EventPreparation):
    def load_events(self):
        super(PAL1EventPreparation,self).load_events()
        self.events.recalled=self.events.correct








