# command line example:
# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=~/scratch/py_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1056M --task=FR1 --workspace-dir=/data10/scratch/mswat/py_run_9 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --python-path=~/RAM_UTILS_GIT

# python ps_report.py --subject=R1086M --task=FR1 --workspace-dir=/data10/scratch/mswat/R1086M_2 --matlab-path=~/eeg --matlab-path=~/matlab/beh_toolbox --matlab-path=~/RAM/RAM_reporting --matlab-path=~/RAM/RAM_sys2Biomarkers --matlab-path=~/RAM_UTILS_GIT/tests/ps2_report/AuxiliaryMatlab --python-path=~/RAM_UTILS_GIT



from ReportUtils import CMLParser,ReportPipeline

cml_parser = CMLParser(arg_count_threshold=1)


cml_parser.arg('--subject','R1250N')
cml_parser.arg('--task','PAL1')
cml_parser.arg('--workspace-dir','D:/scratch_PAL1_report/')
# cml_parser.arg('--workspace-dir','/Users/m/automated_reports/PAL1_reports')
cml_parser.arg('--mount-point','d:/')
# cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')

# cml_parser.arg('--python-path','/Users/m/PTSA_NEW_GIT')



# cml_parser.arg('--subject','R1162N')
# cml_parser.arg('--workspace-dir','/scratch/mswat/automated_reports/PAL1_reports')
# cml_parser.arg('--mount-point','')
# cml_parser.arg('--recompute-on-no-status')
# cml_parser.arg('--exit-on-no-change')

args = cml_parser.parse()


from PAL1EventPreparation import PAL1EventPreparation
from PAL1EventPreparationWithRecall import PAL1EventPreparationWithRecall
from CombinedEventPreparation import CombinedEventPreparation

from ComputePowersWithRecall import ComputePowersWithRecall

from FREventPreparationWithRecall import FREventPreparationWithRecall
from ComputePAL1Powers import ComputePAL1Powers



from ComputePowersWithRecall import ComputePowersWithRecall

from MontagePreparation import MontagePreparation
# from MontagePreparationWithRecall import MontagePreparationWithRecall

from ComputePAL1HFPowers import ComputePAL1HFPowers

from ComputeTTest import ComputeTTest

from ComputeClassifier import ComputeClassifier

from ComputeClassifierWithRecall import ComputeClassifierWithRecall

from ComputeClassifierWithRecall import ComputePAL1Classifier


from ComposeSessionSummary import ComposeSessionSummary

from GenerateReportTasks import *


# turn it into command line options



class Params(object):
    def __init__(self):
        self.width = 5


        self.fr1_start_time = 0.0
        self.fr1_end_time = 1.366
        self.fr1_buf = 1.365

        self.fr1_retrieval_start_time = -0.525
        self.fr1_retrieval_end_time = 0.0
        self.fr1_retrieval_buf = 0.524

        self.pal1_start_time = 0.3
        self.pal1_end_time = 2.0
        self.pal1_buf = 1.2

        self.pal1_retrieval_start_time = -0.625
        self.pal1_retrieval_end_time = -0.1
        self.pal1_retrieval_buf = 0.524


        self.hfs_start_time = 0.4
        self.hfs_end_time = 3.7
        self.hfs_buf = 1.0


        self.encoding_samples_weight = 7.2
        self.pal_samples_weight = 1.93

        self.recall_period = 5.0


        self.filt_order = 4

        self.freqs = np.logspace(np.log10(6), np.log10(180), 8)
        self.hfs = np.logspace(np.log10(2), np.log10(200), 50)
        self.hfs = self.hfs[self.hfs>=70.0]

        self.log_powers = True

        self.penalty_type = 'l2'
        self.C = 7.2e-4

        self.n_perm = 30 # todo change it to 200 in production code


params = Params()

if __name__=='__main__':
    # # sets up processing pipeline
    report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,sessions=args.sessions,
                                     workspace_dir=join(args.workspace_dir, args.subject),
                                     mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
                                     recompute_on_no_status=args.recompute_on_no_status)

    # report_pipeline = ReportPipeline(subject=args.subject, task=args.task,experiment=args.task,
    #                                  workspace_dir=join(args.workspace_dir,args.task+'_'+args.subject), mount_point=args.mount_point, exit_on_no_change=args.exit_on_no_change,
    #                                  recompute_on_no_status=args.recompute_on_no_status)



    # report_pipeline = ReportPipeline(args=args,subject=args.subject,workspace_dir=join(args.workspace_dir, task + '_' + args.subject))


    report_pipeline.add_task(MontagePreparation(params=params, mark_as_completed=False))

    report_pipeline.add_task(PAL1EventPreparation(mark_as_completed=False))

    report_pipeline.add_task(PAL1EventPreparationWithRecall(mark_as_completed=False))

    report_pipeline.add_task(FREventPreparationWithRecall(mark_as_completed=False))

    report_pipeline.add_task(CombinedEventPreparation(mark_as_completed=False))

    report_pipeline.add_task(ComputePAL1Powers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputePowersWithRecall(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputePAL1HFPowers(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeTTest(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputeClassifier(params=params, mark_as_completed=True))

    report_pipeline.add_task(ComputeClassifierWithRecall(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComputePAL1Classifier(params=params, mark_as_completed=False))

    report_pipeline.add_task(ComposeSessionSummary(params=params, mark_as_completed=False))

    report_pipeline.add_task(GeneratePlots(mark_as_completed=False))

    report_pipeline.add_task(GenerateTex(mark_as_completed=False))

    report_pipeline.add_task(GenerateReportPDF(mark_as_completed=False))

    # # report_pipeline.add_task(DeployReportPDF(mark_as_completed=False))

    # starts processing pipeline
    report_pipeline.execute_pipeline()