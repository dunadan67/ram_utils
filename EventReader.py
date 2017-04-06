__author__ = 'm'

import re
from os.path import *
import numpy as np
# from scipy.io import loadmat

from ptsa.data.events import Events
from ptsa.data.rawbinwrapper import RawBinWrapper


from BaseEventReader import BaseEventReader

class EventReader(BaseEventReader):
    def __init__(self, event_file, **kwds):
        BaseEventReader.__init__(self,event_file, **kwds)


    def correct_eegfile_field(self, events):
        events = events[events.eegfile != '[]']  # remove events with no recording
        data_dir_bad = r'/data.*/' + events[0].subject + r'/eeg'
        data_dir_good = r'/data/eeg/' + events[0].subject + r'/eeg'
        for ev in events:
            ev.eegfile = ev.eegfile.replace('eeg.reref', 'eeg.noreref')
            ev.eegfile = re.sub(data_dir_bad, data_dir_good, ev.eegfile)
        return events


    # def read(self):
    #
    #     # calling base class read fcn
    #     evs = BaseEventReader.read(self)
    #
    #
    #     evs = self.correct_eegfile_field(evs)
    #
    #
    #     evs = evs.add_fields(esrc=np.dtype(RawBinWrapper))
    #
    #     import pathlib
    #
    #     eegfiles = np.unique(evs.eegfile)
    #
    #     for eegfile in eegfiles:
    #         eeg_file_path = join(self.data_dir_prefix, str(pathlib.Path(str(eegfile)).parts[1:]))
    #         raw_bin_wrapper = RawBinWrapper(eeg_file_path)
    #         inds = np.where(evs.eegfile == eegfile)[0]
    #         for i in inds:
    #             evs[i]['esrc'] = raw_bin_wrapper
    #
    #     # self.subject_path = str(pathlib.Path(eeg_file_path).parts[:-2])
    #
    #     # attaching
    #     # evs.add_fields(esrc=np.dtype(RawBinWrapper))
    #
    #     self.set_output(evs)
    #
    #     return self.get_output()
    #
    #     # return evs
    #
    #
    #
    #     # for ev in evs:
    #     #     try:
    #     #         eeg_file_path = join(self.data_dir_prefix, str(pathlib.Path(str(ev.eegfile)).parts[1:]))
    #     #         ev.esrc = RawBinWrapper(eeg_file_path)
    #     #         self.raw_data_root=str(eeg_file_path)
    #     #     except TypeError:
    #     #         print 'skipping event with eegfile=',ev.eegfile
    #     #         pass
    #
    #     self.subject_path = str(pathlib.Path(eeg_file_path).parts[:-2])
    #
    #     # attaching
    #     # evs.add_fields(esrc=np.dtype(RawBinWrapper))
    #
    #     self.set_output(evs)
    #
    #     return self.get_output()


    def read(self):

        # calling base class read fcn
        evs = BaseEventReader.read(self)

        # evs = self.correct_eegfile_field(evs) ################## RESTORE THIS

        evs = evs.add_fields(esrc=np.dtype(RawBinWrapper))

        import not_pathlib

        for ev in evs:
            try:
                eeg_file_path = join(self.data_dir_prefix, str(not_pathlib.Path(str(ev.eegfile)).parts[1:]))
                ev.esrc = RawBinWrapper(eeg_file_path)
                self.raw_data_root=str(eeg_file_path)
            except TypeError:
                print 'skipping event with eegfile=',ev.eegfile
                pass

        # self.subject_path = str(pathlib.Path(eeg_file_path).parts[:-2])

        # attaching
        # evs.add_fields(esrc=np.dtype(RawBinWrapper))

        self.set_output(evs)

        return self.get_output()


if __name__=='__main__':

        from EventReader import EventReader
        e_path = join('/Volumes/rhino_root', 'data/events/RAM_FR1/R1060M_events.mat')
        # e_path = '/Users/m/data/events/RAM_FR1/R1056M_events.mat'
        e_reader = EventReader(event_file=e_path, eliminate_events_with_no_eeg=True, data_dir_prefix='/Volumes/rhino_root')

        events = e_reader.read()

        events = e_reader.get_output()