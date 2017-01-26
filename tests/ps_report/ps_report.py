# command line example:

import sys

from setup_utils import parse_command_line, configure_python_paths


from ReportUtils import CMLParser,ReportPipeline
from functools import partial

cml_parser = CMLParser(arg_count_threshold=1)
cml_parser.arg('--task','PS2')
cml_parser.arg('--subject','R1050M')
cml_parser.arg('--workspace-dir','/scratch/leond/ps_reports')
cml_parser.arg('--mount-point','')
#cml_parser.arg('--recompute-on-no-status')

# cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()


from ReportTasks import EventPreparation,JointFR1EventPreparation
from ReportTasks import PSEventPreparation

from ReportTasks import ComputePowers as ComputeFRPowers
from ComputeControlPowers import ComputeControlPowers
from ComputePSPowers import ComputePSPowers

from MontagePreparation import MontagePreparation

from ComputeClassifier import ComputeClassifier

from ComputeControlTable import ComputeControlTable
from ComputePSTable import ComputePSTable

from ComposeSessionSummary import ComposeSessionSummary
from ptsa.data.readers.IndexReader import JsonIndexReader
jr=JsonIndexReader('/protocols/r1.json')

from GenerateReportTasks import *


# turn it into command line options

class Params(object):
    def __init__(self):
        self.width = 5

        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.sham1_start_time = 1.0
        self.sham1_end_time = 2.0
        self.sham_buf = 1.0

        self.sham2_start_time = 10.0 - 3.7
        self.sham2_end_time = 10.0 - 2.7

        self.ps_start_time = -1.0
        self.ps_end_time = 0.0
        self.ps_buf = 1.0
        self.ps_offset = 0.1

        self.filt_order = 4

        self.freqs = np.logspace(np.log10(3), np.log10(180), 8)

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 200

        self.include_fr1 = True
        self.include_catfr1 = True


params = Params()
tasks = jr.experiments(subject=args.subject)
if 'FR1' in tasks and 'catFR1' in tasks:
    FREventPreparation = JointFR1EventPreparation(['FR1','catFR1'])
else:
    FREventPreparation = EventPreparation(tasks=['FR1','catFR1'])
    FREventPreparation.pass_events = partial(EventPreparation.pass_events,FREventPreparation,task='FR')

# sets up processing pipeline

report_pipeline = ReportPipeline(subject=args.subject,
                                 task=args.task,
                                 workspace_dir=join(args.workspace_dir, args.subject),
                                 mount_point=args.mount_point,
                                 exit_on_no_change=args.exit_on_no_change,
                                 recompute_on_no_status=args.recompute_on_no_status)

report_pipeline.add_task(FREventPreparation)

report_pipeline.add_task(PSEventPreparation(tasks=[args.task]))

report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

report_pipeline.add_task(ComputeFRPowers(params=params, mark_as_completed=True,task='FR',name='ComputeFRPowers'))

report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeControlPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputePSPowers(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputeControlTable(params=params, mark_as_completed=True))

report_pipeline.add_task(ComputePSTable(params=params, mark_as_completed=True))

report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

report_pipeline.add_task(GenerateTex(mark_as_completed=False))

report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

# starts processing pipeline
report_pipeline.execute_pipeline()
