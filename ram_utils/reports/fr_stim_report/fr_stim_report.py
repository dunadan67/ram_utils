from os.path import join

from ...ReportUtils import ReportPipeline,ReportSummaryInventory
from ptsa.data.readers import IndexReader

from .FREventPreparation import FREventPreparation
from .EventPreparation import EventPreparation
from .RepetitionRatio import RepetitionRatio
from .ComputeFRPowers import ComputeFRPowers
from .ComputeClassifier import ComputeClassifier
from .EvaluateClassifier import EvaluateClassifier
from .ComputeFRStimPowers import ComputeFRStimPowers
from .MontagePreparation import MontagePreparation
from .ComputeFRStimTable import ComputeFRStimTable
from .ComposeSessionSummary import ComposeSessionSummary
from .GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.ttest_frange = (70.0, 200.0)

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 500

        self.include_fr1 = True
        self.include_catfr1 = True

def run_report(args):

    report_pipeline = build_pipeline(args)

    # starts processing pipeline
    report_pipeline.execute_pipeline()

def run_all_reports(args):
    rsi = ReportSummaryInventory()
    jr = IndexReader.JsonIndexReader(join(args.mount_point,'protocols','r1.json'))
    subjects  = set(jr.subjects(experiment=args.task))
    for subject in subjects:
        montages  = jr.montages(subject=subject,experiment=args.task)
        for montage in montages:
            subject += '_%s'%str(montage) if montage>0 else ''
            args.subject=subject
            report_pipeline= build_pipeline(args)
            report_pipeline.add_task(DeployReportPDF(False))
            report_pipeline.execute_pipeline()
            rsi.add_report_summary(report_pipeline.get_report_summary())

    rsi.output_json_files(args.report_status_dir)

def build_pipeline(args):
    params = Params()
    # sets up processing pipeline
    report_pipeline = ReportPipeline(subject=args.subject, task=args.task, experiment=args.task, sessions=args.sessions,
                                     workspace_dir=join(args.workspace_dir, args.subject), mount_point=args.mount_point,
                                     exit_on_no_change=args.exit_on_no_change,
                                     recompute_on_no_status=args.recompute_on_no_status)
    report_pipeline.add_task(FREventPreparation(mark_as_completed=False))
    report_pipeline.add_task(EventPreparation(mark_as_completed=False))
    report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))
    if 'cat' in args.task:
        report_pipeline.add_task(RepetitionRatio(recompute_all_ratios=True, mark_as_completed=True))
    report_pipeline.add_task(ComputeFRPowers(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComputeFRStimPowers(params=params, mark_as_completed=True))
    report_pipeline.add_task(ComputeFRStimTable(params=params, mark_as_completed=False))
    report_pipeline.add_task(
        EvaluateClassifier(params=params, mark_as_completed=True))  # Note: this IS included in the pdf!
    report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))
    #
    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))
    #
    report_pipeline.add_task(GenerateTex(mark_as_completed=False))
    #
    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))
    return report_pipeline