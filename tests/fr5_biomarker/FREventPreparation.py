__author__ = 'm'

import os
import os.path
import numpy as np

from ptsa.data.readers import BaseEventReader
from ptsa.data.readers.IndexReader import JsonIndexReader

from RamPipeline import *
from ReportUtils import RamTask

import hashlib
from sklearn.externals import joblib
from ReportTasks.RamTaskMethods import create_baseline_events


class FREventPreparation(RamTask):
    def __init__(self, mark_as_completed=True):
        super(FREventPreparation, self).__init__(mark_as_completed)

    def input_hashsum(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        hash_md5 = hashlib.md5()

        fr1_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
        for fname in fr1_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        catfr1_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR1')))
        for fname in catfr1_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        fr3_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR3')))
        for fname in fr3_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        catfr3_event_files = sorted(
            list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='catFR3')))
        for fname in catfr3_event_files:
            with open(fname, 'rb') as f: hash_md5.update(f.read())

        return hash_md5.digest()

    def run(self):
        subject = self.pipeline.subject
        tmp = subject.split('_')
        subj_code = tmp[0]
        montage = 0 if len(tmp) == 1 else int(tmp[1])

        fr1_events_fname = os.path.abspath(
            os.path.join(self.pipeline.mount_point, 'scratch', 'jkragel', 'events_FR5', 'RAM_FR1',
                         subj_code + '_events.mat'))

        json_reader = JsonIndexReader(os.path.join(self.pipeline.mount_point, 'protocols/r1.json'))

        if self.pipeline.args.sessions:
            event_files = [
                json_reader.get_value('task_events', subject=subj_code, montage=montage, experiment='FR1', session=s)
                for s in sorted(self.pipeline.args.sessions)]
        else:
            event_files = sorted(
                list(json_reader.aggregate_values('task_events', subject=subj_code, montage=montage, experiment='FR1')))
        fr1_events = np.concatenate([BaseEventReader(filename=event_path).read() for event_path in event_files]).view(
            np.recarray)

        catfr1_events = np.concatenate([BaseEventReader(filename=event_path).read() for event_path in
                                        json_reader.aggregate_values('task_events', subject=subj_code,
                                                                     experiment='catFR1',
                                                                     montage=montage)]).view(np.recarray)

        # making sure that events have the same columens and do not include stim_params column (which is a composite type
        # that is not used here at all)

        common_column_set = set(list(fr1_events.dtype.names)).intersection(list(catfr1_events.dtype.names))
        common_column_set.remove('stim_params')
        common_column_list = list(common_column_set)

        # col_name_list = list(fr1_events.dtype.names)
        # col_name_list = col_name_list.remove('stim_params')

        catfr1_events = catfr1_events[common_column_list]
        fr1_events = fr1_events[common_column_list]

        # catfr1_events=catfr1_events[list(fr1_events.dtype.names)]
        catfr1_events.session += 100
        fr1_events = np.append(fr1_events, catfr1_events).view(np.recarray)
        fr1_events = create_baseline_events(fr1_events)

        # e_reader = BaseEventReader(filename=fr1_events_fname, eliminate_events_with_no_eeg=True,common_root='scratch')
        # fr1_events = e_reader.read()


        encoding_events_mask = fr1_events.type == 'WORD'
        retrieval_events_mask = (fr1_events.type == 'REC_WORD') | (fr1_events.type == 'REC_BASE')
        irts = np.append([0], np.diff(fr1_events.mstime))
        retrieval_events_mask_0s = retrieval_events_mask & (fr1_events.type == 'REC_BASE')
        retrieval_events_mask_1s = retrieval_events_mask & (fr1_events.type == 'REC_WORD') & (
        fr1_events.intrusion == 0) & (irts > 1000)

        filtered_events = fr1_events[encoding_events_mask | retrieval_events_mask_0s | retrieval_events_mask_1s]

        events = filtered_events.view(np.recarray)

        print len(events), 'WORD events'

        joblib.dump(events, self.get_path_to_resource_in_workspace(subject + '-FR_events.pkl'))
        self.pass_object('FR_events', events)
        # self.pass_object('encoding_events_mask',encoding_events_mask)
        # self.pass_object('retrieval_events_mask_0s',retrieval_events_mask_0s)
        # self.pass_object('retrieval_events_mask_1s',retrieval_events_mask_1s)

    def restore(self):
        subject = self.pipeline.subject
        events = joblib.load(self.get_path_to_resource_in_workspace(subject + '-FR_events.pkl'))
        self.pass_object('FR_events', events)