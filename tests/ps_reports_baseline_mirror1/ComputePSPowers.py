__author__ = 'm'


from RamPipeline import *

import numpy as np
from scipy.stats.mstats import zscore
from morlet import MorletWaveletTransform
from sklearn.externals import joblib

from ptsa.data.events import Events
from ptsa.data.readers import EEGReader


class ComputePSPowers(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params
        self.wavelet_transform = MorletWaveletTransform()

    def restore(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        ps_pow_mat_pre = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_pow_mat_pre.pkl'))
        ps_pow_mat_post = joblib.load(self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_pow_mat_post.pkl'))

        self.pass_object('ps_pow_mat_pre',ps_pow_mat_pre)
        self.pass_object('ps_pow_mat_post',ps_pow_mat_post)


    def run(self):
        subject = self.pipeline.subject
        experiment = self.pipeline.experiment

        #fetching objects from other tasks
        events = self.get_passed_object(self.pipeline.experiment+'_events')
        # channels = self.get_passed_object('channels')
        # tal_info = self.get_passed_object('tal_info')
        monopolar_channels = self.get_passed_object('monopolar_channels')
        bipolar_pairs = self.get_passed_object('bipolar_pairs')


        sessions = np.unique(events.session)
        print experiment, 'sessions:', sessions

        ps_pow_mat_pre, ps_pow_mat_post = self.compute_ps_powers(events, sessions, monopolar_channels, bipolar_pairs, experiment)

        joblib.dump(ps_pow_mat_pre, self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_pow_mat_pre.pkl'))
        joblib.dump(ps_pow_mat_post, self.get_path_to_resource_in_workspace(subject+'-'+experiment+'-ps_pow_mat_post.pkl'))

        self.pass_object('ps_pow_mat_pre',ps_pow_mat_pre)
        self.pass_object('ps_pow_mat_post',ps_pow_mat_post)

    def compute_ps_powers(self, events, sessions, monopolar_channels, bipolar_pairs, experiment):
        n_freqs = len(self.params.freqs)
        n_bps = len(bipolar_pairs)

        pow_mat_pre = pow_mat_post = None

        pow_ev = None
        samplerate = winsize = bufsize = None

        monopolar_channels_list = list(monopolar_channels)
        for sess in sessions:
            sess_events = events[events.session == sess]
            # print type(sess_events)

            n_events = len(sess_events)

            print 'Loading EEG for', n_events, 'events of session', sess

            pre_start_time = self.params.ps_start_time - self.params.ps_offset
            pre_end_time = self.params.ps_end_time - self.params.ps_offset

            # eegs_pre = Events(sess_events).get_data(channels=channels, start_time=pre_start_time, end_time=pre_end_time,
            #             buffer_time=self.params.ps_buf, eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)


            eeg_pre_reader = EEGReader(events=sess_events, channels=np.array(monopolar_channels_list),
                                   start_time=pre_start_time,
                                   end_time=pre_end_time, buffer_time=self.params.ps_buf)

            eegs_pre = eeg_pre_reader.read()


            if samplerate is None:
                # samplerate = round(eegs_pre.samplerate)
                # samplerate = eegs_pre.attrs['samplerate']

                samplerate = float(eegs_pre['samplerate'])

                winsize = int(round(samplerate*(pre_end_time-pre_start_time+2*self.params.ps_buf)))
                bufsize = int(round(samplerate*self.params.ps_buf))
                print 'samplerate =', samplerate, 'winsize =', winsize, 'bufsize =', bufsize
                pow_ev = np.empty(shape=n_freqs*winsize, dtype=float)
                self.wavelet_transform.init(self.params.width, self.params.freqs[0], self.params.freqs[-1], n_freqs, samplerate, winsize)

            # mirroring
            nb_ = int(round(samplerate*(self.params.ps_buf)))
            eegs_pre[...,-nb_:] = eegs_pre[...,-nb_-2:-2*nb_-2:-1]

            dim3_pre = eegs_pre.shape[2]  # because post-stim time inreval does not align for all stim events (stims have different duration)
                                          # we have to take care of aligning eegs_post ourselves time dim to dim3

            # eegs_post = np.zeros_like(eegs_pre)

            from ptsa.data.TimeSeriesX import TimeSeriesX
            eegs_post = TimeSeriesX(np.zeros_like(eegs_pre),dims=eegs_pre.dims,coords=eegs_pre.coords)


            post_start_time = self.params.ps_offset
            post_end_time = self.params.ps_offset + (self.params.ps_end_time - self.params.ps_start_time)
            for i_ev in xrange(n_events):
                ev_offset = sess_events[i_ev].pulse_duration
                if ev_offset > 0:
                    if experiment == 'PS3' and sess_events[i_ev].nBursts > 0:
                        ev_offset *= sess_events[i_ev].nBursts + 1
                    ev_offset *= 0.001
                else:
                    ev_offset = 0.0

                # eeg_post = Events(sess_events[i_ev:i_ev+1]).get_data(channels=channels, start_time=post_start_time+ev_offset,
                #             end_time=post_end_time+ev_offset, buffer_time=self.params.ps_buf,
                #             eoffset='eegoffset', keep_buffer=True, eoffset_in_time=False)

                eeg_post_reader = EEGReader(events=sess_events[i_ev:i_ev+1], channels=np.array(monopolar_channels_list),
                                       start_time=post_start_time+ev_offset,
                                       end_time=post_end_time+ev_offset, buffer_time=self.params.ps_buf)

                eeg_post = eeg_post_reader.read()

                dim3_post = eeg_post.shape[2]
                # here we take care of possible mismatch of time dim length
                if dim3_pre == dim3_post:
                    eegs_post[:,i_ev:i_ev+1,:] = eeg_post
                elif dim3_pre < dim3_post:
                    eegs_post[:,i_ev:i_ev+1,:] = eeg_post[:,:,:-1]
                else:
                    eegs_post[:,i_ev:i_ev+1,:-1] = eeg_post

            # mirroring
            eegs_post[...,:nb_] = eegs_post[...,2*nb_:nb_:-1]

            print 'Computing', experiment, 'powers'

            sess_pow_mat_pre = np.empty(shape=(n_events, n_bps, n_freqs), dtype=np.float)
            sess_pow_mat_post = np.empty_like(sess_pow_mat_pre)

            for i,ti in enumerate(bipolar_pairs):
                bp = ti['channel_str']
                print 'Computing powers for bipolar pair', bp
                elec1 = np.where(monopolar_channels == bp[0])[0][0]
                elec2 = np.where(monopolar_channels == bp[1])[0][0]

            #
            # for i,ti in enumerate(tal_info):
            #     bp = ti['channel_str']
            #     print 'Computing powers for bipolar pair', bp
            #     elec1 = np.where(channels == bp[0])[0][0]
            #     elec2 = np.where(channels == bp[1])[0][0]


                bp_data_pre = eegs_pre[elec1] - eegs_pre[elec2]
                # bp_data_pre.attrs['samplerate'] = samplerate

                bp_data_pre = bp_data_pre.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    #pow_pre_ev = phase_pow_multi(self.params.freqs, bp_data_pre[ev], to_return='power')
                    self.wavelet_transform.multiphasevec(bp_data_pre[ev][0:winsize], pow_ev)
                    #sess_pow_mat_pre[ev,i,:] = np.mean(pow_pre_ev[:,nb_:-nb_], axis=1)
                    pow_ev_stripped = np.reshape(pow_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]
                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat_pre[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

                bp_data_post = eegs_post[elec1] - eegs_post[elec2]
                # bp_data_post.attrs['samplerate'] = samplerate

                bp_data_post = bp_data_post.filtered([58,62], filt_type='stop', order=self.params.filt_order)
                for ev in xrange(n_events):
                    #pow_post_ev = phase_pow_multi(self.params.freqs, bp_data_post[ev], to_return='power')
                    self.wavelet_transform.multiphasevec(bp_data_post[ev][0:winsize], pow_ev)
                    #sess_pow_mat_post[ev,i,:] = np.mean(pow_post_ev[:,nb_:-nb_], axis=1)
                    pow_ev_stripped = np.reshape(pow_ev, (n_freqs,winsize))[:,bufsize:winsize-bufsize]
                    if self.params.log_powers:
                        np.log10(pow_ev_stripped, out=pow_ev_stripped)
                    sess_pow_mat_post[ev,i,:] = np.nanmean(pow_ev_stripped, axis=1)

            sess_pow_mat_pre = sess_pow_mat_pre.reshape((n_events, n_bps*n_freqs))
            #sess_pow_mat_pre = zscore(sess_pow_mat_pre, axis=0, ddof=1)

            sess_pow_mat_post = sess_pow_mat_post.reshape((n_events, n_bps*n_freqs))
            #sess_pow_mat_post = zscore(sess_pow_mat_post, axis=0, ddof=1)

            sess_pow_mat_joint = zscore(np.vstack((sess_pow_mat_pre,sess_pow_mat_post)), axis=0, ddof=1)
            sess_pow_mat_pre = sess_pow_mat_joint[:n_events,...]
            sess_pow_mat_post = sess_pow_mat_joint[n_events:,...]

            pow_mat_pre = np.vstack((pow_mat_pre,sess_pow_mat_pre)) if pow_mat_pre is not None else sess_pow_mat_pre
            pow_mat_post = np.vstack((pow_mat_post,sess_pow_mat_post)) if pow_mat_post is not None else sess_pow_mat_post

        return pow_mat_pre, pow_mat_post
