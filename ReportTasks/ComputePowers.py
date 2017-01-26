from RamPipeline import *

import numpy as np
import time

from ptsa.data.filters import MorletWaveletFilterCpp,MonopolarToBipolarMapper
from ptsa.extensions.morlet.morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.readers import EEGReader
from ptsa.data.readers.IndexReader import JsonIndexReader
from ReportUtils import ReportRamTask

import hashlib
from scipy.stats.mstats import zscore


def normalize_sessions(pow_mat, events):
    sessions = np.unique(events.session)
    for sess in sessions:
        sess_event_mask = (events.session == sess)
        pow_mat[sess_event_mask] = zscore(pow_mat[sess_event_mask], axis=0, ddof=1)
    return pow_mat

class ComputePowers(ReportRamTask):
    def __init__(self, params, task,mark_as_completed=True,name=None):
        super(ComputePowers, self).__init__(mark_as_completed,name=name)
        self.params = params
        self.pow_mat = None
        self.samplerate = None
        self.wavelet_transform = MorletWaveletTransform()
        self.task = task


    def compute_powers(self, events, sessions, monopolar_channels, bipolar_pairs):
        n_bps = len(bipolar_pairs)
        n_freqs = len(self.params.freqs)
        eeg_reader = EEGReader(events=events, channels=monopolar_channels,
                               start_time=self.params.fr1_start_time, end_time=self.params.fr1_end_time,
                               buffer_time=0.0)

        eegs = eeg_reader.read().add_mirror_buffer(duration=self.params.fr1_buf)

        eegs = MonopolarToBipolarMapper(time_series=eegs,bipolar_pairs=bipolar_pairs).filter()
        print 'eegs.shape:',eegs.shape
        print 'len(freqs)',n_freqs
        pow_mat,phase_mat = MorletWaveletFilterCpp(eegs, freqs=self.params.freqs, output='power',cpus=10).filter()
        del phase_mat
        if pow_mat is None:
            exit()
        pow_mat = pow_mat.transpose('events','bipolar_pairs','frequency','time').remove_buffer(duration=self.params.fr1_buf)
        if self.params.log_powers:
            pow_mat = np.log10(pow_mat)

        pow_mat = np.nanmean(pow_mat,-1).reshape(len(events),n_freqs*n_bps)

        self.pow_mat = normalize_sessions(pow_mat, events)

    def get_events(self):
        task = self.task
        return self.get_passed_object(task + '_events')

    def pass_objects(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        self.pass_object(task + '_pow_mat', self.pow_mat)
        self.pass_object('samplerate', self.samplerate)

        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-pow_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task

        self.pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-pow_mat.pkl'))
        self.samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

        self.pass_objects()

    def run(self):
        events = self.get_events()
        sessions = np.unique(events.session)
        print 'sessions for %s:'%self.task, sessions

        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')
        tic=time.clock()
        self.compute_powers(events, sessions, monopolar_channels, bipolar_pairs)
        toc=time.clock()
        print '%f seconds elapsed'%(toc-tic)

        self.pass_objects()


class ComputePowersOld(ComputePowers):
    def compute_powers(self, events, sessions, monopolar_channels, bipolar_pairs):
        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        pow_ev = None
        winsize = bufsize = None
        for sess in sessions:
            sess_events = events[events.session == sess]
            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            eeg_reader = EEGReader(events=sess_events, channels=monopolar_channels,
                                   start_time=self.params.fr1_start_time,
                                   end_time=self.params.fr1_end_time, buffer_time=0.0)

            eegs = eeg_reader.read()

            if eeg_reader.removed_bad_data():
                print 'REMOVED SOME BAD EVENTS !!!'
                sess_events = eegs['events'].values.view(np.recarray)
                n_events = len(sess_events)
                events = np.hstack((events[events.session != sess], sess_events)).view(np.recarray)
                ev_order = np.argsort(events, order=('session', 'list', 'mstime'))
                events = events[ev_order]
                self.pass_object(self.pipeline.task + '_events', events)

            eegs = eegs.add_mirror_buffer(duration=self.params.fr1_buf)

            if self.samplerate is None:
                self.samplerate = float(eegs.samplerate)
                winsize = int(round(
                    self.samplerate * (self.params.fr1_end_time - self.params.fr1_start_time + 2 * self.params.fr1_buf)))
                bufsize = int(round(self.samplerate * self.params.fr1_buf))
                print 'samplerate =', self.samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                pow_ev = np.empty(shape=n_freqs * winsize, dtype=float)
                self.wavelet_transform.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs,
                                            self.samplerate, winsize)

            print 'Computing powers'

            sess_pow_mat = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)

            for i, bp in enumerate(bipolar_pairs):

                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

                bp_data = np.subtract(eegs[elec1], eegs[elec2])
                bp_data.attrs['samplerate'] = self.samplerate

                bp_data = bp_data.filtered([58, 62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    self.wavelet_transform.multiphasevec(bp_data[ev][0:winsize], pow_ev)
                    pow_ev_stripped = np.reshape(pow_ev, (n_freqs, winsize))[:, bufsize:winsize - bufsize]
                    pow_zeros = np.where(pow_ev_stripped == 0.0)[0]
                    if len(pow_zeros) > 0:
                        print 'zero powers:', bp, ev
                        print sess_events[ev].eegfile, sess_events[ev].eegoffset
                        if len(pow_zeros) > 0:
                            print bp, ev
                            print sess_events[ev].eegfile, sess_events[ev].eegoffset
                            self.raise_and_log_report_exception(
                                exception_type='NumericalError',
                                exception_message='Corrupt EEG File'
                            )
                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat[ev, i, :] = np.nanmean(pow_ev_stripped, axis=1)

            self.pow_mat = np.concatenate((self.pow_mat, sess_pow_mat),
                                          axis=0) if self.pow_mat is not None else sess_pow_mat

        self.pow_mat = normalize_sessions(np.reshape(self.pow_mat, (len(events), n_bps * n_freqs)),events)


class ComputeHFPowers(ComputePowers):
    def restore(self):
        subject = self.pipeline.subject
        task=self.task

        pow_mat = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-hf_pow_mat.pkl'))
        samplerate = joblib.load(self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))
        self.pass_object('hf_pow_mat', pow_mat)
        self.pass_object('hf_samplerate', samplerate)


    def pass_objects(self):
        subject = self.pipeline.subject
        task=self.pipeline.task
        self.pass_object('hf_pow_mat', self.pow_mat)
        self.pass_object('hf_samplerate', self.samplerate)

        joblib.dump(self.pow_mat, self.get_path_to_resource_in_workspace(subject + '-' + task + '-hf_pow_mat.pkl'))
        joblib.dump(self.samplerate, self.get_path_to_resource_in_workspace(subject + '-samplerate.pkl'))

    def compute_powers(self, events, sessions, monopolar_channels, bipolar_pairs):
        n_bps = len(bipolar_pairs)
        n_freqs = len(self.params.freqs)
        super(ComputeHFPowers,self).compute_powers(events,sessions,monopolar_channels,bipolar_pairs)
        self.pow_mat = np.reshape(self.pow_mat,(len(events),n_bps,n_freqs))
        self.pow_mat = np.nanmean(self.pow_mat,2)
        print 'pow_mat.shape',self.pow_mat.shape


