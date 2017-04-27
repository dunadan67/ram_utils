# -*- coding: utf-8 -*-

from RamPipeline import *

from PlotUtils import PlotData, BarPlotData, PlotDataCollection, PanelPlot

import numpy as np
import datetime
from subprocess import call

from ReportUtils import ReportRamTask
from TextTemplateUtils import replace_template,replace_template_to_string
from TexUtils.latex_table import latex_table

class GeneratePlots(ReportRamTask):
    def __init__(self):
        super(GeneratePlots,self).__init__(mark_as_completed=False)



    def run(self):
        self.create_dir_in_workspace('reports')
        task = self.pipeline.task
        subject= self.pipeline.subject


        ps_events = self.get_passed_object('ps_events')
        ps_sessions = np.unique(ps_events.session)
        ps4_session_summaries = self.get_passed_object('ps_session_summary')
        # ps4_session_summaries = False
        if ps4_session_summaries:
            for session in ps_sessions:

                session_summary = ps4_session_summaries[session]
                xmin = min([amps.min() for amps in session_summary.amplitudes])
                xmax = max([amps.max() for amps in session_summary.amplitudes])
                ymin = min([dels.min() for dels in session_summary.delta_classifiers])
                ymax = max([dels.max() for dels in session_summary.delta_classifiers])

                panel_plot  = PanelPlot(i_max = 1, j_max = 2, xfigsize = 18.0, yfigsize = 14.0,labelsize=15)
                panel_plot.add_plot_data(0,0,x=session_summary.amplitudes[0],y=session_summary.delta_classifiers[0],
                                         xlim=(xmin-50,xmax+50),ylim = (ymin-0.05,ymax+0.05),
                                         linestyle='',marker='x',color='blue', xlabel=session_summary.locations[0],
                                         xlabel_fontsize = 18, ylabel = 'Change in classifier output (post - pre)',ylabel_fontsize = 24)
                panel_plot.add_plot_data(0,1,x=session_summary.amplitudes[1], y=session_summary.delta_classifiers[1],
                                         xlim=(xmin-50, xmax+50), ylim=(ymin - 0.05, ymax + 0.05),
                                         linestyle = '',marker='x',color='blue',xlabel=session_summary.locations[1],
                                         xlabel_fontsize=18,)
                plt = panel_plot.generate_plot()
                session_summary.PS_PLOT_FILE = self.get_path_to_resource_in_workspace('reports','PS4_%d_dc_plot.pdf'%session)
                plt.savefig(session_summary.PS_PLOT_FILE,dpi=300)


        fr5_events = self.get_passed_object(task+'_events')
        fr5_session_summaries = self.get_passed_object('fr_session_summary')
        if fr5_session_summaries:

            xval_output = self.get_passed_object(task+'_xval_output')
            fr1_summary = xval_output[-1]

            panel_plot = PanelPlot(xfigsize=15, yfigsize=7.5, i_max=1, j_max=2, labelsize=16, wspace=5.0)

            pd1 = PlotData(x=fr1_summary.fpr, y=fr1_summary.tpr, xlim=[0.0, 1.0], ylim=[0.0, 1.0],
                           xlabel='False Alarm Rate\n(a)', ylabel='Hit Rate', xlabel_fontsize=20, ylabel_fontsize=20,
                           levelline=((0.0, 1.0), (0.0, 1.0)), color='k', markersize=1.0)

            pc_diff_from_mean = (
            fr1_summary.low_pc_diff_from_mean, fr1_summary.mid_pc_diff_from_mean, fr1_summary.high_pc_diff_from_mean)

            ylim = np.max(np.abs(pc_diff_from_mean)) + 5.0
            if ylim > 100.0:
                ylim = 100.0
            pd2 = BarPlotData(x=(0, 1, 2), y=pc_diff_from_mean, ylim=[-ylim, ylim],
                              xlabel='Tercile of Classifier Estimate\n(b)', ylabel='Recall Change From Mean (%)',
                              x_tick_labels=['Low', 'Middle', 'High'], xlabel_fontsize=20, ylabel_fontsize=20,
                              xhline_pos=0.0, barcolors=['grey', 'grey', 'grey'], barwidth=0.5)

            panel_plot.add_plot_data(0, 0, plot_data=pd1)
            panel_plot.add_plot_data(0, 1, plot_data=pd2)

            plot = panel_plot.generate_plot()

            plot_out_fname = self.get_path_to_resource_in_workspace(
                'reports/' + self.pipeline.subject + '-roc_and_terc_plot.pdf')

            plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')

            self.pass_object('ROC_AND_TERC_PLOT_FILE',plot_out_fname)

            sessions = np.unique(fr5_events.session)

            serial_positions = np.arange(1, 13)
            for session_summary in fr5_session_summaries:

                # P_REC and PFR
                panel_plot = PanelPlot(i_max=1,j_max=2,xfigsize=15, yfigsize=7.5, title='', labelsize=18)

                pdca = PlotDataCollection(xlim=(0,12), ylim=(0.0, 1.0), xlabel='(a)', ylabel='Stim vs Non-Stim Items', xlabel_fontsize=20,ylabel_fontsize=20)
                pd1a = PlotData(x=serial_positions, y=session_summary.prob_stim_recall,linestyle='-',label='Stim')
                pdca.add_plot_data(pd1a)
                pd2a = PlotData(x = serial_positions, y=session_summary.prob_nostim_recall,linestyle = '--',label='No Stim')
                pdca.add_plot_data(pd2a)
                panel_plot.add_plot_data_collection(0,0,plot_data_collection=pdca)

                pdcb = PlotDataCollection(xlim=(0,12), ylim=(0.0, 1.0), xlabel='(b)', ylabel='', xlabel_fontsize=20,ylabel_fontsize=20)
                pd1b = PlotData(x=serial_positions, y = session_summary.prob_first_stim_recall, linestyle = '-', label = 'Stim')
                pd2b = PlotData(x=serial_positions, y=session_summary.prob_first_nostim_recall, linestyle = '--', label = 'No Stim')
                pdcb.add_plot_data(pd1b)
                pdcb.add_plot_data(pd2b)
                panel_plot.add_plot_data_collection(0,1,plot_data_collection=pdcb)
                plot = panel_plot.generate_plot()
                plot.legend()
                plot_out_fname = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-split_prob_recall_plot_' + session_summary.STIMTAG + '_' + str(session_summary.frequency) + '.pdf')
                plot.savefig(plot_out_fname, dpi=300, bboxinches='tight')
                session_summary.PROB_RECALL_PLOT_FILE = plot_out_fname


                # Change in recall

                panel_plot = PanelPlot(xfigsize=6, yfigsize=7.5, i_max=1, j_max=1, title='', labelsize=18)

                ylim = np.max(np.abs(session_summary.pc_diff_from_mean)) + 5.0
                if ylim > 100.0:
                    ylim = 100.0
                pd = BarPlotData(x=(0,1), y=session_summary.pc_diff_from_mean, ylim=[-ylim,ylim], xlabel='Items', ylabel='% Recall Difference (Stim-NoStim)', x_tick_labels=['Stim', 'PostStim'], xhline_pos=0.0, barcolors=['grey', 'grey'], xlabel_fontsize=18, ylabel_fontsize=18, barwidth=0.5)
                panel_plot.add_plot_data(0, 0, plot_data=pd)

                plot = panel_plot.generate_plot()

                session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE = self.get_path_to_resource_in_workspace('reports/' + task + '-' + subject + '-stim_vs_non_stim_halves_plot_' + session_summary.STIMTAG + '_' + str(session_summary.frequency) + '.pdf')

                plot.savefig(session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE, dpi=300, bboxinches='tight')


                # number of stims and number of recalls

                n_lists = len(session_summary.n_stims_per_list)

                xfigsize = 7 * n_lists / 25.0
                if xfigsize < 10.0:
                    xfigsize = 10.0
                elif xfigsize > 18.0:
                    xfigsize = 18.0
                panel_plot = PanelPlot(xfigsize=xfigsize, yfigsize=10.0, i_max=1, j_max=1, title='', xlabel='List',
                                       ylabel='# of items', labelsize=20)

                pdc = PlotDataCollection()
                pdc.xlabel = 'List number'
                pdc.xlabel_fontsize = 20
                pdc.ylabel = '#items'
                pdc.ylabel_fontsize = 20

                x_tick_labels = np.array([str(k) for k in session_summary.list_number])
                x_tick_labels[1::5] = ''
                x_tick_labels[2::5] = ''
                x_tick_labels[3::5] = ''
                x_tick_labels[4::5] = ''

                bpd_1 = BarPlotData(x=np.arange(n_lists), y=session_summary.n_stims_per_list,
                                    x_tick_labels=x_tick_labels, title='', alpha=0.3)
                stim_x = np.where(session_summary.is_stim_list)[0]
                stim_y = session_summary.n_recalls_per_list[session_summary.is_stim_list]
                pd_1 = PlotData(x=stim_x, y=stim_y, ylim=(0, 12),
                                title='', linestyle='', color='red', marker='o', markersize=12)
                plot_data_list = [bpd_1,pd_1]
                if session_summary.is_stim_list.all():
                    nostim_x = np.where(session_summary.is_ps_list)[0]
                    nostim_y = session_summary.n_recalls_per_list[session_summary.is_ps_list]
                    pd_2 = PlotData(x=nostim_x, y=nostim_y, ylim=(0, 12),
                                    title='', linestyle='', color='grey', marker='o', markersize=12)
                    baseline_x = np.where(session_summary.is_baseline_list)[0]
                    baseline_y = session_summary.n_recalls_per_list[np.array(session_summary.is_baseline_list)]
                    pd_3 = PlotData(x=baseline_x, y=baseline_y, ylim=(0, 12),
                                    title='', linestyle='', color='blue', marker='o', markersize=12)
                    plot_data_list.extend([pd_2,pd_3])
                else:
                    nostim_x = np.where(session_summary.is_nonstim_list)[0]
                    nostim_y = session_summary.n_recalls_per_list[session_summary.is_nonstim_list]
                    pd_2 = PlotData(x=nostim_x, y=nostim_y, ylim=(0, 12),
                                    title='', linestyle='', color='blue', marker='o', markersize=12)
                    plot_data_list.append(pd_2)

                for pd in plot_data_list:
                    if (pd.x.shape and pd.y.shape) and all(pd.x.shape) and all(pd.y.shape):
                        print pd.x.shape
                        print pd.y.shape
                        pdc.add_plot_data(pd)

                # for i in xrange(len(session_summary.list_number) - 1):
                #     if session_summary.list_number[i] > session_summary.list_number[i + 1]:
                #         sep_pos = i + 0.5
                #         sep_plot_data = PlotData(x=[0], y=[0], levelline=[[sep_pos, sep_pos], [0, 12]], color='white',
                #                                  alpha=0.0)
                #         pdc.add_plot_data(sep_plot_data)

                panel_plot.add_plot_data_collection(0, 0, plot_data_collection=pdc)

                plot = panel_plot.generate_plot()

                session_summary.STIM_AND_RECALL_PLOT_FILE = self.get_path_to_resource_in_workspace(
                    'reports/' + task + '-' + subject + '-stim_and_recall_plot_' + session_summary.STIMTAG + '-' + str(
                        session_summary.frequency) + '.pdf')

                plot.savefig(session_summary.STIM_AND_RECALL_PLOT_FILE, dpi=300, bboxinches='tight')

                panel_plot = PanelPlot(xfigsize=8,yfigsize=5,i_max=1,j_max=1)
                pd = PlotData(x=range(1,len(session_summary.prob_stim)+1),y=session_summary.prob_stim,ylim=[0,1],label_size=18,
                              xlabel='Serial Position',ylabel='Probability of stim',color='black')
                panel_plot.add_plot_data(0,0,plot_data=pd)
                plot = panel_plot.generate_plot()
                session_summary.PROB_STIM_PLOT_FILE = self.get_path_to_resource_in_workspace('reports/'+subject+'p_stim_plot_'+session_summary.STIMTAG+'-'+str(session_summary.frequency)+'.pdf')
                plot.savefig(session_summary.PROB_STIM_PLOT_FILE,dpi=300,bboxinches='tight')


class GenerateTex(ReportRamTask):
    def run(self):

        subject = self.pipeline.subject
        experiment = self.pipeline.task
        date = datetime.date.today()

        ps_latex = self.generate_ps_latex()
        fr5_latex = self.generate_fr5_latex()

        replace_template('ps4_fr5_report_base.tex.tpl',self.get_path_to_resource_in_workspace('reports','ps4_fr5_report.tex'),
                         {
                             '<SUBJECT>':subject,
                             '<EXPERIMENT>':experiment,
                             '<DATE>':date,
                             '<PS4_SECTION>':ps_latex,
                             '<FR5_SECTION>':fr5_latex})
        self.pass_object('report_tex_file_name','ps4_fr5_report.tex')


    def generate_ps_latex(self):
        ps_events = self.get_passed_object('ps_events')
        ps_session_summary = self.get_passed_object('ps_session_summary')
        ps_latex = ''
        if ps_session_summary:
            n_sessions = len(ps_session_summary)
            n_electrodes = len(self.get_passed_object('monopolar_channels'))
            if ps_events is not None and ps_events.shape:
                for session in np.unique(ps_events.session):
                    if ps_session_summary[session].preferred_location:
                        session_decision = replace_template_to_string('ps_decision.tex.tpl',
                                                                      {
                                                                          '<PREFERRED_LOCATION>': ps_session_summary[
                                                                              session].preferred_location,
                                                                          '<PREFERRED_AMPLITUDE>': ps_session_summary[
                                                                              session].preferred_amplitude,
                                                                          '<TSTAT>': ps_session_summary[session].tstat,
                                                                          '<PVALUE>': ps_session_summary[session].pvalue
                                                                      })
                    else:
                        session_decision = 'No significant parameters found.'
                    ps_session_latex = replace_template_to_string('ps4_session.tex.tpl',
                                                                  {
                                                                      '<SESSION>':session,
                                                                      '<LOC1>': ps_session_summary[session].locations[0],
                                                                      '<LOC2>': ps_session_summary[session].locations[1],
                                                                      '<PS_PLOT_FILE>': ps_session_summary[session].PS_PLOT_FILE,
                                                                      '<DECISION>': session_decision
                                                                  })
                    ps_latex += ps_session_latex
                ps_latex = replace_template_to_string('PS4_section.tex.tpl',
                                                      {
                                                          '<SUBJECT>':self.pipeline.subject,
                                                          '<NUMBER_OF_PS4_SESSIONS>':n_sessions,
                                                          '<NUMBER_OF_ELECTRODES>':n_electrodes,
                                                          '<SESSION_DATA>':latex_table(self.get_passed_object('ps_session_data')),
                                                          '<PS4_SESSION_PAGES>':ps_latex
                                                      })
        return ps_latex

    def generate_fr5_latex(self):
        subject =self.pipeline.subject
        task = self.pipeline.task
        monopolar_channels = self.get_passed_object('monopolar_channels')
        xval_output = self.get_passed_object(task+'_xval_output')
        fr1_xval_output = self.get_passed_object('xval_output')
        fr5_auc = xval_output[-1].auc
        fr5_perm_pvalue = self.get_passed_object(task+'_pvalue')
        fr1_auc = fr1_xval_output[-1].auc
        fr1_pvalue = self.get_passed_object('pvalue')
        session_data =self.get_passed_object('fr5_session_table')


        fr5_events  = self.get_passed_object(task+'_events')


        n_sessions = len(np.unique(fr5_events.session))

        fr5_session_summary = self.get_passed_object('fr_session_summary')
        all_session_tex = ''

        if fr5_events is not None and all(fr5_events.shape):

            for session_summary in fr5_session_summary:
                sessions = session_summary.sessions
                if len(sessions)>1:
                    sessions = ','.join(sessions)
                else:
                    sessions = str(sessions)

                biomarker_tex = replace_template_to_string('biomarker_plots.tex.tpl',
                                                           {'<STIM_VS_NON_STIM_HALVES_PLOT_FILE>':session_summary.STIM_VS_NON_STIM_HALVES_PLOT_FILE})

                recognition_tex = (replace_template_to_string('recognition.tex.tpl',
                                                             {
                                                                 '<PHITS_STIM>': '%2.2d' % session_summary.pc_hits_stim,
                                                                 '<PHITS_NO_STIM>': '%2.2d' % session_summary.pc_hits_nostim,
                                                                 '<PFALSE_ALARMS>': '%2.2d' % session_summary.pc_false_alarms,
                                                                 '<DPRIME>': '%2.2d' % session_summary.dprime,

                                                             })
                                   if session_summary.dprime != -999 else '')
                item_level_comparison = '' #if session_summary.chisqr_last == -999 else latex_table(session_summary.last_recall_table)
                session_tex = replace_template_to_string('fr5_session.tex.tpl',
                             {
                                 '<SESSIONS>':          sessions,
                                 '<STIMTAG>':           session_summary.stimtag,
                                 '<REGION>':            session_summary.region_of_interest,
                                 '<AMPLITUDE>':         session_summary.amplitude,
                                 '<N_WORDS>':           session_summary.n_words,
                                 '<N_CORRECT_WORDS>':   session_summary.n_correct_words,
                                 '<PC_CORRECT_WORDS>':  '%2.2f'%session_summary.pc_correct_words,
                                 '<N_PLI>':             session_summary.n_pli,
                                 '<PC_PLI>':            '%2.2f'%session_summary.pc_pli,
                                 '<N_ELI>':             session_summary.n_eli,
                                 '<PC_ELI>':           '%2.2f'% session_summary.pc_eli,
                                 '<N_MATH>':            session_summary.n_math,
                                 '<N_CORRECT_MATH>':session_summary.n_correct_math,
                                 '<PC_CORRECT_MATH>':'%2.2f'%session_summary.pc_correct_math,
                                 '<MATH_PER_LIST>':'%2.2f'%session_summary.math_per_list,
                                 '<PROB_RECALL_PLOT_FILE>':session_summary.PROB_RECALL_PLOT_FILE,
                                 '<N_CORRECT_STIM>':session_summary.n_correct_stim,
                                 '<N_TOTAL_STIM>':session_summary.n_total_stim,
                                 '<N_CORRECT_NONSTIM>':session_summary.n_correct_nonstim,
                                 '<N_TOTAL_NONSTIM>':session_summary.n_total_nonstim,
                                 '<PC_FROM_STIM>':'%2.2f'%session_summary.pc_from_stim,
                                 '<PC_FROM_NONSTIM>':'%2.2f'%session_summary.pc_from_nonstim,
                                 '<COMPARISON_LIST_TYPE>': 'non-stim' if ((fr5_events[np.in1d(fr5_events.session,session_summary.sessions)].phase=='NON-STIM')).any() else 'FR1',
                                 '<ITEMLEVEL_COMPARISON>': item_level_comparison,
                                 '<CHISQR>':'%.4f'%session_summary.chisqr,
                                 '<PVALUE>':'%.4f'%session_summary.pvalue,
                                 '<N_STIM_INTR>': session_summary.n_stim_intr,
                                 '<PC_FROM_STIM_INTR>':'%2.2f'%session_summary.pc_from_stim_intr,
                                 '<N_NONSTIM_INTR>':session_summary.n_nonstim_intr,
                                 '<PC_FROM_NONSTIM_INTR>':'%2.2f'%session_summary.pc_from_nonstim_intr,
                                 '<STIM_AND_RECALL_PLOT_FILE>':session_summary.STIM_AND_RECALL_PLOT_FILE,
                                 '<PROB_STIM_PLOT_FILE>':session_summary.PROB_STIM_PLOT_FILE,
                                 '<BIOMARKER_PLOTS>':biomarker_tex,
                                 '<RECOGNITION>':recognition_tex,
                             }
                )
                all_session_tex += session_tex
        fr5_tex = replace_template_to_string(
            'FR5_section.tex.tpl',
            {
                '<SUBJECT>':subject,
                '<NUMBER_OF_ELECTRODES>':len(monopolar_channels),
                '<NUMBER_OF_SESSIONS>':n_sessions,
                '<AUC>':'%2.2f'%fr1_auc,
                '<PERM-P-VALUE>':fr1_pvalue if fr1_pvalue>0 else '<0.01',
                '<SESSION_DATA>':latex_table(session_data),
                '<FR5-AUC>':'%2.2f'%fr5_auc,
                '<FR5-PERM-P-VALUE>':fr5_perm_pvalue if fr5_perm_pvalue>0 else '<0.01',
                '<ROC_AND_TERC_PLOT_FILE>':self.get_passed_object('ROC_AND_TERC_PLOT_FILE'),
                '<REPORT_PAGES>':all_session_tex
            }
        )
        return fr5_tex


class GenerateReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=False):
        super(GenerateReportPDF,self).__init__(mark_as_completed)

    def run(self):
        output_directory = self.get_path_to_resource_in_workspace('reports')

        texinputs_set_str = r'export TEXINPUTS="' + output_directory + '":$TEXINPUTS;'

        report_tex_file_name = self.get_passed_object('report_tex_file_name')

        pdflatex_command_str = texinputs_set_str + 'module load Tex; pdflatex' \
                               + ' -output-directory '+output_directory\
                               + ' -shell-escape ' \
                               + self.get_path_to_resource_in_workspace('reports/'+report_tex_file_name)

    # + 'module load Tex; pdflatex '\

        call([pdflatex_command_str], shell=True)

        report_core_file_name, ext = splitext(report_tex_file_name)
        report_file = join(output_directory,report_core_file_name+'.pdf')
        self.pass_object('report_file',report_file)


class DeployReportPDF(ReportRamTask):
    def __init__(self, mark_as_completed=True):
        super(DeployReportPDF,self).__init__(mark_as_completed)

    def run(self):
        report_file = self.get_passed_object('report_file')
        self.pipeline.deploy_report(report_path=report_file)













