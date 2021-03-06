from RamPipeline import *

import numpy as np
from scipy.stats import ttest_ind
from sklearn.externals import joblib

from scipy.stats import describe

import normalize


class ComputeTTest(RamTask):
    def __init__(self, params, mark_as_completed=True):
        RamTask.__init__(self, mark_as_completed)
        self.params = params

    def run(self):
        print 'Computing t-stats'

        subject = self.pipeline.subject
        task = self.pipeline.task

        pow_mat = self.get_passed_object('pow_mat')

        freq_sel = np.tile((self.params.freqs>=self.params.ttest_frange[0]) & (self.params.freqs<=self.params.ttest_frange[1]), pow_mat.shape[1] / self.params.freqs.size)
        pow_mat = pow_mat[:,freq_sel]

        print 'Power Matrix stats:'
        print describe(pow_mat, axis=None, ddof=1)

        #pow_mat = np.mean(pow_mat, axis=(2,3))

        events = self.get_passed_object(self.pipeline.task+'_events')
        sessions = np.unique(events.session)

        # norm_func = normalize.standardize_pow_mat if self.params.norm_method=='zscore' else normalize.normalize_pow_mat
        # pow_mat = norm_func(pow_mat, events, sessions)[0]

        self.ttest = {}
        for sess in sessions:
            sel = (events.session==sess)
            sess_events = events[sel]

            sess_pow_mat = pow_mat[sel,:]

            sess_recalls = np.array(sess_events.recalled, dtype=np.bool)

            recalled_sess_pow_mat = sess_pow_mat[sess_recalls,:]
            nonrecalled_sess_pow_mat = sess_pow_mat[~sess_recalls,:]

            t,p = ttest_ind(recalled_sess_pow_mat, nonrecalled_sess_pow_mat, axis=0)
            self.ttest[sess] = (t,p)

        recalls = np.array(events.recalled, dtype=np.bool)

        recalled_pow_mat = pow_mat[recalls,:]
        nonrecalled_pow_mat = pow_mat[~recalls,:]

        t,p = ttest_ind(recalled_pow_mat, nonrecalled_pow_mat, axis=0)
        self.ttest[-1] = (t,p)

        self.pass_object('ttest', self.ttest)
        joblib.dump(self.ttest, self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))

    def restore(self):
        subject = self.pipeline.subject
        task = self.pipeline.task
        self.ttest = joblib.load(self.get_path_to_resource_in_workspace(subject + '-' + task + '-ttest.pkl'))
        self.pass_object('ttest', self.ttest)
