import sys
import time
import traceback

import numpy as np
import dill

import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from astropy import units as u
from pbjam.star import star as PbjamStar
from corner import corner

from . import DipoleStar
from .theta import ThetaReg
from .reggae import reggae, UNITS

from zeta.utils import α_to_q
from zeta.plots import period_echelle_power_plot, make_τ

FIELDS = list(ThetaReg.__match_args__) + ['norm']
BOUNDS = ThetaReg.bounds + [[20, 100]]

STYLE = {
    -1: dict(marker='v'),
    0: dict(marker='o'),
    1: dict(marker='^'),
    'Manual': dict(facecolor="C0"),
    'Simplex': dict(facecolor="C1"),
    'Diff-Evol': dict(facecolor="C2"),
}

plt.rcParams['backend'] = 'QtAgg'

class ReggaeDebugWindow(QtWidgets.QMainWindow):

    def __init__(self, input=None, session=None):

        qapp = QtWidgets.QApplication.instance()
        self.qapp = qapp
        if not qapp:
            qapp = QtWidgets.QApplication(sys.argv)

        super().__init__()
        self.init_ui()

        if isinstance(input, DipoleStar):
            self.load_reggae(reggae=input)
        elif isinstance(input, PbjamStar):
            self.load_pbjam(pbjam=input)

        if session is not None:
            self.load(session)

        self.show()
        self.activateWindow()
        self.raise_()
        qapp.exec()

    def init_ui(self):

        self._main = QtWidgets.QSplitter()
        self.setCentralWidget(self._main)
        left = QtWidgets.QTabWidget()
        right = QtWidgets.QWidget()
        self._main.addWidget(left)
        self._main.addWidget(right)

        #--- LEFT PANE ---

        self._ax = {}
        self._curves = {}

        ## Data tab: Plots of TS and PS

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        left.addTab(widget, "Data Plots")

        ts_canvas = FigureCanvas(Figure())
        layout.addWidget(NavigationToolbar(ts_canvas, self))
        layout.addWidget(ts_canvas)
        self._ax['ts'] = ts_canvas.figure.subplots()

        ps_canvas = FigureCanvas(Figure())
        layout.addWidget(ps_canvas)
        layout.addWidget(NavigationToolbar(ps_canvas, self))
        self._ax['ps'] = ps_canvas.figure.subplots()

        ## χ2 tab: local cost-function landscape (for local optimisation)

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)
        left.addTab(widget, "χ² landscape")

        for i, field in enumerate(FIELDS):
            chi2_canvas = FigureCanvas(Figure())
            layout.addWidget(chi2_canvas, i//3, i%3)
            self._ax[f'chi2_{i}'] = chi2_canvas.figure.subplots()
            self._ax[f'chi2_{i}'].set_xlabel(DipoleStar.labels[i])

        ## Echelle tab: Frequency and Period Echelle

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        left.addTab(widget, "Échelle Diagrams")

        canvas = FigureCanvas(Figure())
        layout.addWidget(NavigationToolbar(canvas, self))
        layout.addWidget(canvas)
        self._ax['echelle'] = canvas.figure.subplots()
        self._echelle_points = {}

        canvas = FigureCanvas(Figure())
        layout.addWidget(canvas)
        layout.addWidget(NavigationToolbar(canvas, self))
        self._ax['period_echelle'] = canvas.figure.subplots()
        self._period_echelle_points = {}

        ## MCMC tab: Corner Plot

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        left.addTab(widget, "MCMC")

        canvas = FigureCanvas(Figure())
        layout.addWidget(NavigationToolbar(canvas, self))
        layout.addWidget(canvas)
        # this is a FIGURE OBJECT, NOT an Axes object
        self._ax['MCMC'] = canvas.figure

        #--- RIGHT PANE ---

        layout_right = QtWidgets.QVBoxLayout(right)
        tabs = QtWidgets.QTabWidget()
        layout_right.addWidget(tabs)

        groupbox = QtWidgets.QWidget()
        tabs.addTab(groupbox, "Reggae Parameters")
        layout = QtWidgets.QFormLayout(groupbox)
        spinboxes = {}

        for field in FIELDS:
            edit = QtWidgets.QDoubleSpinBox()
            edit.setSingleStep(.01)
            edit.setDecimals(5)
            edit.setValue(1)
            edit.setMinimum(-1e5)
            edit.setMaximum(1e5)
            spinboxes[field] = edit
            layout.addRow(f"{field}", edit)

        # set some reasonable default values
        spinboxes['epsilon_g'].setValue(.5)
        spinboxes['alpha_g'].setValue(0)
        spinboxes['d01'].setValue(0)
        spinboxes['log_omega_rot'].setValue(-100)
        spinboxes['inclination'].setValue(np.pi/4)

        pair  = QtWidgets.QHBoxLayout()

        textbox = QtWidgets.QDoubleSpinBox()
        textbox.setSingleStep(1)
        textbox.setDecimals(0)
        textbox.setValue(100)
        textbox.setMinimum(1)
        textbox.setMaximum(5000)
        spinboxes['opt_iters'] = textbox
        pair.addWidget(textbox)

        button = QtWidgets.QPushButton("Optimize", None)
        button.setStatusTip(f"Optimize Reggae parameters around current θ_reg with simplex algorithm")
        button.clicked.connect(self.optimize)
        pair.addWidget(button)
        layout.addRow("Simplex Iterations", pair)

        self.checkboxes = {}

        checkbox = QtWidgets.QCheckBox("Show Simplex points in échelle diagrams", None)
        self.checkboxes['show_opt'] = checkbox
        layout.addRow(checkbox)

        checkbox = QtWidgets.QCheckBox("Show Diff-Evol points in échelle diagrams", None)
        self.checkboxes['show_de'] = checkbox
        layout.addRow(checkbox)

        pair  = QtWidgets.QHBoxLayout()
        checkbox = QtWidgets.QCheckBox("Show m = ±1 in échelle diagrams", None)
        self.checkboxes['m'] = checkbox
        pair.addWidget(checkbox)

        button = QtWidgets.QPushButton("Recompute echelle points", None)
        button.setStatusTip(f"Recompute points on echelle diagram")
        button.clicked.connect(self.recompute)
        pair.addWidget(button)

        layout.addRow(pair)

        pair  = QtWidgets.QHBoxLayout()

        edit = QtWidgets.QDoubleSpinBox()
        edit.setSingleStep(.01)
        edit.setDecimals(5)
        edit.setValue(1)
        edit.setMinimum(0)
        spinboxes['q'] = edit
        pair.addWidget(edit)

        button = QtWidgets.QPushButton("Auto-q", None)
        button.setStatusTip(f"Compute an effective value of q from the provided coupling parameters")
        button.clicked.connect(lambda *x: self.spinboxes['q'].setValue(self.get_q()))
        pair.addWidget(button)

        layout.addRow(f"Effective q", pair)

        pair  = QtWidgets.QHBoxLayout()

        for _ in 'x0', 'x1':
            textbox = QtWidgets.QDoubleSpinBox()
            textbox.setSingleStep(.1)
            textbox.setDecimals(5)
            textbox.setValue(0)
            textbox.setMinimum(-1)
            textbox.setMaximum(2)
            spinboxes[_] = textbox
            pair.addWidget(textbox)
        spinboxes['x1'].setValue(1)
        layout.addRow("x-limits (ΔΠ)",  pair)

        pair  = QtWidgets.QHBoxLayout()
        for _ in 'y0', 'y1':
            textbox = QtWidgets.QDoubleSpinBox()
            textbox.setSingleStep(.1)
            textbox.setDecimals(7)
            textbox.setValue(0)
            textbox.setMinimum(0)
            textbox.setMaximum(5000)
            spinboxes[_] = textbox
            pair.addWidget(textbox)
        layout.addRow("y-limits (μHz)", pair)
        
        button = QtWidgets.QPushButton("Recompute period-echelle", None)
        button.setStatusTip(f"Recompute stretched echelle power plot (expensive)")
        button.clicked.connect(self.period_echelle_power_plot)
        layout.addRow(button)

        self.spinboxes = spinboxes

        ## DEBUG: comment this out if not needed

        self.debug = QtWidgets.QPlainTextEdit()
        self.debug.setFont(QtGui.QFont("Monospace"))
        layout.addRow(self.debug)

        def debug():
            command = self.debug.document().toPlainText()
            exec(command)

        button = QtWidgets.QPushButton("Debug", None)
        button.clicked.connect(debug)
        layout.addRow(button)

        ## END DEBUG


        groupbox = QtWidgets.QWidget()
        tabs.addTab(groupbox, "MCMC Bounds")
        layout = QtWidgets.QFormLayout(groupbox)

        bounds = {}

        for i, field in enumerate(FIELDS):
            pair = QtWidgets.QHBoxLayout()
            for j in [0, 1]:
                textbox = QtWidgets.QDoubleSpinBox()
                textbox.setSingleStep(.01)
                textbox.setDecimals(5)
                textbox.setMinimum(-1e5)
                textbox.setMaximum(1e5)
                textbox.setValue(BOUNDS[i][j])
                bounds[f"{field}_{j}"] = textbox

                pair.addWidget(textbox)

            # button to sweep

            sweep = QtWidgets.QPushButton("⊳", None)
            sweep.clicked.connect((lambda j: (lambda *_: self.sweep(j)))(i))
            sweep.setStatusTip(f"Perform parameter sweep for {field} over specified bounds")
            pair.addWidget(sweep)

            layout.addRow(f"{field}", pair)

        self.bounds = bounds
        button = QtWidgets.QPushButton("Recompute from Current θ_reg", None)
        button.clicked.connect(self.newbounds)
        layout.addRow(button)

        n_g_lims = {}

        pair  = QtWidgets.QHBoxLayout()
        for _ in 'upper', 'lower':
            textbox = QtWidgets.QDoubleSpinBox()
            textbox.setSingleStep(1)
            textbox.setDecimals(0)
            textbox.setValue(0)
            textbox.setMinimum(0)
            textbox.setMaximum(1e5)
            n_g_lims[_] = textbox
            pair.addWidget(textbox)

        self.n_g_lims = n_g_lims

        button = QtWidgets.QPushButton("Auto", None)
        button.setStatusTip(f"Auto-evaluate n_g range from provided MCMC bounds")
        button.clicked.connect(self.update_n_g)
        pair.addWidget(button)

        layout.addRow("n_g limits", pair)

        pair  = QtWidgets.QHBoxLayout()

        textbox = QtWidgets.QDoubleSpinBox()
        textbox.setSingleStep(1)
        textbox.setDecimals(0)
        textbox.setValue(100)
        textbox.setMinimum(1)
        textbox.setMaximum(5000)
        spinboxes['de_iters'] = textbox
        pair.addWidget(textbox)

        button = QtWidgets.QPushButton("Run", None)
        button.setStatusTip(f"Optimize Reggae parameters with genetic (differential-evolution) algorithm")
        button.clicked.connect(self.de)
        pair.addWidget(button)

        layout.addRow("Diff-evol Iterations", pair)

        textbox = QtWidgets.QDoubleSpinBox()
        textbox.setSingleStep(.01)
        textbox.setDecimals(5)
        textbox.setValue(1/200)
        textbox.setMinimum(0)
        textbox.setMaximum(1)
        spinboxes['lw'] = textbox
        layout.addRow("PSD Model Linewidth (Δν)", textbox)

        textbox = QtWidgets.QDoubleSpinBox()
        textbox.setSingleStep(1)
        textbox.setDecimals(3)
        textbox.setValue(1)
        textbox.setMinimum(0)
        textbox.setMaximum(100)
        spinboxes['soften'] = textbox
        layout.addRow("Likelihood softening factor", textbox)

        pair  = QtWidgets.QHBoxLayout()

        checkbox = QtWidgets.QCheckBox("Use dynamic nested sampling", None)
        self.checkboxes['dynamic'] = checkbox
        pair.addWidget(checkbox)

        button = QtWidgets.QPushButton("Run Dynesty", None)
        button.setStatusTip(f"Optimize Reggae parameters with nested sampling")
        button.clicked.connect(self.dynesty)
        pair.addWidget(button)
        layout.addRow(pair)


        # Console

        groupbox = QtWidgets.QGroupBox("Console")
        layout_right.addWidget(groupbox)
        layout = QtWidgets.QFormLayout(groupbox)

        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QtGui.QFont("Monospace"))
        layout.addWidget(self.console)

        progress = QtWidgets.QProgressBar()
        layout_right.addWidget(progress)
        self.progress = progress
        self.threadpool = QtCore.QThreadPool()

        #--- TOOLBAR ---

        toolbar = QtWidgets.QToolBar("Actions")
        self.addToolBar(QtCore.Qt.ToolBarArea.RightToolBarArea, toolbar)

        button_action = QtGui.QAction("Import PBJam", self)
        button_action.setStatusTip("Import existing PBJam star object from pickle")
        button_action.triggered.connect(self.load_pbjam)
        toolbar.addAction(button_action)

        button_action = QtGui.QAction("Load Reggae", self)
        button_action.setStatusTip("Load existing reggae object from pickle")
        button_action.triggered.connect(self.load_reggae)
        toolbar.addAction(button_action)

        button_action = QtGui.QAction("Load Session", self)
        button_action.setStatusTip("Load session variables from pickle")
        button_action.triggered.connect(self.load)
        toolbar.addAction(button_action)

        button_action = QtGui.QAction("Dump Session", self)
        button_action.setStatusTip("Dump current session to a pickle file")
        button_action.triggered.connect(self.dump)
        toolbar.addAction(button_action)

        button_action = QtGui.QAction("Dump Reggae", self)
        button_action.setStatusTip("Dump current reggae object to a pickle file")
        button_action.triggered.connect(self.dump_reggae)
        toolbar.addAction(button_action)

        self.print("Initialised Reggae Console.")

    def tqdm(self, gen, total=None, **kwargs):
        if total is None:
            total = len(gen)
        for i, _ in enumerate(gen):
            self.progress.setValue(int(i/total*100))
            yield _
        self.progress.setValue(100)

    def sweep(self, i):
        '''
        Parameter sweep over given bounds
        '''

        field = FIELDS[i]

        bounds = [self.bounds[f"{field}_{_}"].value() for _ in (0, 1)]
        self.print(f"Performing parameter sweep for {field} between {bounds[0]} and {bounds[1]}...")
        x = np.linspace(*bounds, 100)
        θreg, norm = self.get_state()
        θ = np.array([*θreg.asarray(), norm])

        def job():
            return np.array([-self.reggae.ln_like([*θ[:i], _, *θ[i+1:]]) for _ in self.tqdm(x)])

        # def cleanup(chi2):

        chi2 = job()

        ax = self._ax[f"chi2_{i}"]
        ax.clear()
        ax.plot(x, chi2)
        ax.set_xlabel(DipoleStar.labels[i])
        ax.figure.canvas.draw()

        # set state to minimum of sweep

        self.spinboxes[f'{field}'].setValue(x[np.argmin(chi2)])
        self.recompute()

        # worker = Worker(job)
        # worker.signals.result.connect(cleanup)
        # self.threadpool.start(worker)

    def recompute(self):
        self.sync_state()
        θreg, norm = self.get_state()
        self.replot(θreg, norm, label='Manual')

        if self.checkboxes['show_opt'].isChecked():
            try:
                x = self.reggae.simplex_results['x']
                θreg, norm = ThetaReg(*x[:-1]), x[-1]
                self.replot(θreg, norm, label="Simplex")
            except AttributeError:
                pass
        else:
            for d in self._echelle_points, self._curves, self._period_echelle_points:
                for _ in list(d.keys()):
                    if 'Simplex' in _:
                        d[_].remove()
                        del d[_]

        if self.checkboxes['show_de'].isChecked():
            try:
                u = self.reggae.de_results[0][0]
                x = self.reggae.ptform(u)
                θreg, norm = ThetaReg(*x[:-1]), x[-1]
                self.replot(θreg, norm, label="Diff-Evol")

            except AttributeError:
                pass
        else:
            for d in self._echelle_points, self._curves, self._period_echelle_points:
                for _ in list(d.keys()):
                    if 'Diff-Evol' in _:
                        d[_].remove()
                        del d[_]


    def replot(self, θreg, norm, label='Manual'):
        '''
        regenerate echelle diagrams
        '''
        ax = self._ax['echelle']
        ylims = ax.get_ylim()
        dnu = 10**self.reggae.theta_asy.log_dnu

        nu1, zeta = self.reggae.l1model.getl1(self.reggae.theta_asy, θreg)
        m = (nu1 < np.max(self.reggae.f + dnu))&(nu1>np.min(self.reggae.f))
        sizes = np.power(1-zeta[m], 2/3) * 40

        if self.checkboxes['m'].isChecked():
            labels = [f"{label}{_}" for _ in ('_+', '_-', '')]
        else:
            labels = [label]
            for _ in ('+', '-'):
                for d in self._echelle_points, self._period_echelle_points:
                    if f"{label}_{_}" in d:
                        d[f"{label}_{_}"].remove()
                        del d[f"{label}_{_}"]

        δν_rot = 10.**θreg.log_omega_rot / reggae.nu_to_omega

        for l in labels:
            mm = (1 if '_+' in l else (-1 if '_-' in l else 0))
            nu = nu1[m] + mm * zeta[m] * δν_rot
            if l not in self._echelle_points:
                self._echelle_points[l] = ax.scatter(nu % dnu, (nu // dnu) * dnu,
                                                            s=sizes, edgecolor='white',
                                                            label=None if mm else label,
                                                            **STYLE[label], **STYLE[mm])
                
            else:
                # do update stuff
                temp = np.array([nu % dnu, (nu // dnu) * dnu]).T
                self._echelle_points[l].set_offsets(temp)
                self._echelle_points[l].set_sizes(sizes)

        ax.set_xlim(0, dnu)
        ax.set_ylim(*ylims)
        ax.legend()
        ax.figure.canvas.draw()

        ax = self._ax['ps']
        y = self.reggae.model([*θreg.asarray(), norm])
        if label not in self._curves:
            self._curves[label] = ax.plot(self.reggae.f, y, label=label)[0]
            ax.legend()
        else:
            self._curves[label].set_ydata(y)
        ax.figure.canvas.draw()

        # we only replot the period echelle diagram if the period echelle power plot has already been generated

        ax = self._ax['period_echelle']
        if not len(ax.collections):
            return

        sizes = np.power(zeta[m], 2/3) * 40

        ν_p = self.get_ν_p()
        ΔΠ1 = self.get_ΔΠ()
        q = self.spinboxes['q'].value()
        Δν = 10.**self.reggae.theta_asy.log_dnu
        τ_ = make_τ(q, ν_p, Δν, ΔΠ1, principal=False)

        for l in labels:
            mm = (1 if '_+' in l else (-1 if '_-' in l else 0))
            nu = nu1[m] + mm * zeta[m] * δν_rot

            # replication
            x = np.concatenate([(τ_(nu) / ΔΠ1) % 1 + _ for _ in (-1, 0, 1)])
            y = np.concatenate([nu for _ in (-1, 0, 1)])
            s = np.concatenate([sizes for _ in (-1, 0, 1)])

            if l not in self._period_echelle_points:
                self._period_echelle_points[l] = ax.scatter(x, y,
                                                            s=s, edgecolor='white',
                                                            label=None if mm else label,
                                                            **STYLE[label], **STYLE[mm])
            else:
                temp = np.array([x, y]).T
                self._period_echelle_points[l].set_offsets(temp)
                self._period_echelle_points[l].set_sizes(s)

        ax.set_xlim([self.spinboxes[f'x{_}'].value() for _ in (0, 1)])
        ax.set_ylim([self.spinboxes[f'y{_}'].value() for _ in (0, 1)])
        ax.legend()
        ax.figure.canvas.draw()


    def load_pickle(self, description):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, description, ".", "Pickle files (*.pkl)")
        if fname[0] != '':
            with open(fname[0], 'rb') as f:
                o = dill.load(f)
            return o
        else:
            return None

    def write_pickle(self, obj, description):
        fname = QtWidgets.QFileDialog.getSaveFileName(self, description, ".", "Pickle files (*.pkl)")
        if fname[0] != '':
            with open(fname[0], 'wb') as f:
                dill.dump(obj, f)

    def load_pbjam(self, *, pbjam=None):
        if pbjam is None:
            pbjam = self.load_pickle("Open PBJam pickle file")
        if pbjam is not None:
            try:
                self.reggae = DipoleStar(pbjam)
                self.load_reggae_actions()
                self.print(f"Loaded pbjam star with ID {self.reggae.star.ID}")
            except Exception:
                self.print("Invalid pickle contents")

    def load_reggae(self, *, reggae=None):
        if reggae is None:
            reggae = self.load_pickle("Open Reggae pickle file")
        if reggae is not None:
            try:
                assert(isinstance(reggae, DipoleStar))
                self.reggae = reggae
                self.load_reggae_actions()
                self.print(f"Loaded reggae object with ID {self.reggae.star.ID}")
            except Exception:
                self.print("Invalid pickle contents")

    def load_reggae_actions(self):

        # populate diagnostic plots

        self._curves = {}

        ax = self._ax['ps']
        ax.clear()
        ax.plot(self.reggae.star.f, self.reggae.star.s, c='black', label="Original SNR")
        ax.plot(self.reggae.f, self.reggae.s, label=r"Divided by $\ell = 0, 2$")
        ax.set_ylabel(r"SNR")
        ax.set_xlabel(r"$\nu/\mu$Hz")
        ax.legend()
        ax.figure.canvas.draw()

        ax = self._ax['echelle']
        try:
            ax.images[-1].colorbar.remove()
        except IndexError:
            pass
        ax.clear()
        seis = self.reggae.pg.to_seismology() # astropy seismology object
        numax = 10**self.reggae.theta_asy.log_numax
        dnu = 10**self.reggae.theta_asy.log_dnu
        try:
            ax = seis.plot_echelle(deltanu=dnu * u.uHz, numax=numax * u.uHz, ax=ax)
            ax.set_aspect('auto')
        except ImportError:
            print("Could not draw colourbar without importing different backend")
        ax.figure.canvas.draw()

        ax = self._ax['period_echelle']
        try:
            ax.collections[-1].colorbar.remove()
        except IndexError:
            pass
        ax.clear()
        ax.figure.canvas.draw()

        self.plot_MCMC()

        # populate textboxes

        self.n_g_lims['lower'].setValue(self.reggae.l1model.n_g[-1])
        self.n_g_lims['upper'].setValue(self.reggae.l1model.n_g[0])

        theta_asy = self.reggae.theta_asy
        dnu = 10.**(theta_asy.log_dnu)
        numax = 10.**(theta_asy.log_numax)
        n = self.reggae.l1model.n_orders // 2 + 1

        self.spinboxes['y0'].setValue(max(self.reggae.f[0], dnu, numax-n*dnu))
        self.spinboxes['y1'].setValue(self.reggae.f[-1])

    def update_n_g(self):
        self.sync_state()
        self.reggae.l1model.n_g = self.reggae.select_n_g()
        self.n_g_lims['lower'].setValue(self.reggae.l1model.n_g[-1])
        self.n_g_lims['upper'].setValue(self.reggae.l1model.n_g[0])

    def get_ΔΠ(self):
        θ_reg = self.get_state()[0]
        return θ_reg.dPi0 / np.sqrt(2) * UNITS['DPI0']

    def get_q(self):
        θ_reg = self.get_state()[0]
        θ_asy = self.reggae.theta_asy

        Δν = np.power(10, θ_asy.log_dnu)
        νmax = np.power(10, θ_asy.log_numax)

        ω = νmax / 1e6 * 2 * np.pi

        ΔΠ1 = self.get_ΔΠ()
        α = (UNITS['P_L'] * θ_reg.p_L + UNITS['P_D'] * θ_reg.p_D) * ω**2

        return(α_to_q(α, Δν, ΔΠ1, νmax/1e6))

    def get_ν_p(self):
        # from definition in reggae class
        
        theta_asy = self.reggae.theta_asy
        theta_reg = self.get_state()[0]

        dnu = 10.**(theta_asy.log_dnu)
        nu_0 = theta_asy.nu_0(self.reggae.l1model.n_orders)
        d02 = 10.**(theta_asy.log_d02)
        d01 = theta_reg.d01
        nu_1 = nu_0 + dnu / 2 - d02/3 + d01 * dnu

        return nu_1

    def period_echelle_power_plot(self):

        ν_p = self.get_ν_p()
        ΔΠ1 = self.get_ΔΠ()
        q = self.spinboxes['q'].value()

        ax = self._ax['period_echelle']
        try:
            ax.images[-1].colorbar.remove()
        except IndexError:
            pass
        ax.clear()
        self._period_echelle_points = {}

        mask = (self.reggae.f > self.spinboxes['y0'].value()) & (self.reggae.f < self.spinboxes['y1'].value())

        period_echelle_power_plot(self.reggae.f[mask], self.reggae.s[mask], ν_p, ΔΠ1, q, ax=ax,
            x0=self.spinboxes['x0'].value(), x1=self.spinboxes['x1'].value(),
            vmin=np.median(self.reggae.s), vmax=self.spinboxes['norm'].value())

        ax.set_xlabel(r"$(\tau /\Delta\Pi_1) \mod 1$")
        ax.set_ylabel(r"$\nu/\mu$Hz")
        ax.figure.canvas.draw()

    def sync_state(self):
        '''
        Synchronise UI with state of reggae object
        '''

        # 1. n_g bounds

        start = self.n_g_lims['lower'].value()
        stop = self.n_g_lims['upper'].value()
        self.reggae.l1model.n_g = np.arange(start, stop+1)[::-1]

        # 2. PSD Model options

        self.reggae.l1model.lw = self.spinboxes['lw'].value()
        self.reggae.soften = self.spinboxes['soften'].value()

        # 3. MCMC bounds

        self.reggae.bounds = self.get_bounds()

    def load(self, *, session=None):
        if session is None:
            session = self.load_pickle("Load session variables from pickle file")
        if session is not None:
            try:
                spinboxes = session['spinboxes']
                bounds = session['bounds']
                n_g_lims = session['n_g']

                # update GUI

                for _ in spinboxes:
                    self.spinboxes[_].setValue(spinboxes[_])

                for i, field in enumerate(FIELDS):
                    for _ in range(2):
                        self.bounds[f"{field}_{_}"].setValue(bounds[i][_])

                for i, _ in enumerate(['lower', 'upper']):
                    self.n_g_lims[_].setValue(n_g_lims[i])

                self.print(f"Loaded session")
                self.sync_state()
            except Exception as e:
                self.print("Failed to load session: expected object to be dict.")
                self.print(f"{str(e)}")

    def print(self, text):
        self.console.appendPlainText(f"{text}")
        self.statusBar().showMessage(f"{text}")

    def optimize(self, maxiter=1000):
        maxiter = self.spinboxes['opt_iters'].value()
        gen = self.tqdm(range(int(maxiter)))

        def callback(*args, **kwargs):
            for _ in gen:
                return

        self.print("Beginning Simplex Optimisation…")
        self.reggae.simplex(*self.get_state(), method="Nelder-Mead", options=dict(maxiter=maxiter), callback=callback)
        self.progress.setValue(100)
        self.print("Complete.")

    def de(self):
        self.sync_state()
        self.print("Beginning Differential-Evolution Optimisation…")
        self.reggae.genetic_algorithm(solve_kwargs=dict(tqdm=self.tqdm), maxiters=self.spinboxes['de_iters'].value())
        self.progress.setValue(100)
        self.print("Complete.")

    def newbounds(self):
        self.sync_state()
        θ_reg, norm = self.get_state()

        # only update bounds for certain quantities

        self.bounds[f"dPi0_0"].setValue(θ_reg.dPi0 / 1.1)
        self.bounds[f"dPi0_1"].setValue(θ_reg.dPi0 * 1.1)

        self.bounds[f"log_omega_rot_0"].setValue(θ_reg.log_omega_rot - .5)
        self.bounds[f"log_omega_rot_1"].setValue(θ_reg.log_omega_rot + .5)
        self.sync_state()

    def dynesty(self):
        self.sync_state()
        self.print("Beginning Dynesty run…")
        self.dynesty_result = self.reggae(dynamic=self.checkboxes['dynamic'].isChecked())
        self.print("Complete.")
        self.plot_MCMC()

    def get_state(self):
        return ThetaReg(**{_: self.spinboxes[_].value() for _ in FIELDS if _ != 'norm'}), self.spinboxes['norm'].value()

    def get_bounds(self):
        return [tuple(self.bounds[f"{field}_{_}"].value() for _ in (0, 1)) for field in FIELDS]

    def dump(self):
        spinboxes = {_: self.spinboxes[_].value() for _ in self.spinboxes}
        bounds = self.get_bounds()
        n_g = tuple(self.n_g_lims[_].value() for _ in ['lower', 'upper'])

        self.write_pickle({
            'spinboxes': spinboxes, 'bounds': bounds, 'n_g': n_g
            }, 'Save session variables to pickle file')

    def dump_reggae(self):
        self.write_pickle(self.reggae, "Save reggae object to pickle file")

    def plot_MCMC(self):
        sampler = getattr(self.reggae, "sampler", None)
        if sampler is None:
            return

        self.dynesty_result = self.reggae.summarise()
        samples = self.dynesty_result['new_samples']

        fig = self._ax['MCMC']
        corner(samples, labels=DipoleStar.labels, fig=fig)
        fig.canvas.draw()

    def closeEvent(self, *args):
        ...


class Signals(QtCore.QObject):
    error = QtCore.pyqtSignal(tuple)
    finished = QtCore.pyqtSignal()
    result = QtCore.pyqtSignal(object)

class Worker(QtCore.QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = Signals()

    @QtCore.pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(
                *self.args, **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done
