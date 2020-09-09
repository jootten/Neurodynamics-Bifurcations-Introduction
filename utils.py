from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.animation as animation

x0 = 0.9
r = 2.9

def step_seq(r, x_n):
    return r * x_n * (1 - x_n)

def logistic_interactive():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    tdata = np.arange(30)
    xdata = []
    xn = x0
    for i in range(30):
            xdata.append(xn)
            xn = step_seq(r, xn)
    line = Line2D(tdata, xdata, marker="o", linestyle="-", color="r")
    ax.add_line(line)
    ax.grid()
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)

    ax_x0 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_r = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_x0 = Slider(ax_x0, "Initial size $x_0$", 0, 1, valinit=x0)
    slider_r = Slider(ax_r, "Bifurcation Parameter r", 0, 4, valinit=r)


    def update(val):
        xdata = []
        xn = slider_x0.val
        r = slider_r.val
        for i in range(30):
            xdata.append(xn)
            xn = step_seq(r, xn)
        line.set_ydata(xdata)
        fig.canvas.draw_idle()

    slider_x0.on_changed(update)
    slider_r.on_changed(update)

    plt.show()

    
def logistic(x, r):
    return x * r * (1 - x)

    
def logistic_map():
    
    # Generate list values -- iterate for each value of r
    #for r in [i * spacing for i in range(int(1/spacing),int(4/spacing))]:
    #   rlist.append(r) 
    #   xlist.append(iterate(randint(iter-res/2,iter+res/2), seed, r))
    n = 10000
    r = np.linspace(2, 4, n)
    x = 1e-5 * np.ones(n)
    plt.subplots(figsize=(10, 7))
    plt.title("Logistic Map")
    for i in range(int(n / 10)):
        x = logistic(x, r)
        if i >= (int(n / 10 - 100)):
            plt.plot(r, x, ",k", alpha=.1, color='white')


    plt.xlim(1.9, 4.1)
    plt.ylim(-0.1,1.1)
    plt.show()
    
import numpy as np
import scipy.optimize
import sys

# solver for NaKModel
class NaKModel:
    
    def __init__(self, V0, n0, delta=.001, fast_K=False, steep_Na=False, low_threshold_K=False):
        # Injected current
        self.I = 0
        # Initialize state variables
        self.V = V0
        self.n = n0
        self.Vdot = 0
        self.ndot = 0
        # Set step size
        self.delta = delta
        # Initialize constants depending on neuron
        # Steady state IV relation
        if steep_Na:
            self.V_v_half = -30
            self.V_k = 7
        else:
            self.V_v_half = -20
            self.V_k = 15
        # Steady state n
        self.n_k = 5
        if low_threshold_K:
            self.n_v_half = -45
        else:
            self.n_v_half = -25
        # Kinetics n
        if fast_K:
            self.tau = 0.152
        else:
            self.tau = 1
        # Neurophysiological properties
        if low_threshold_K:
            self.E_L = -78
        else:
            self.E_L = -80
        self.E_Na = 60
        self.E_K = -90
        self.g_L = 8
        if steep_Na:
            self.g_L = 1
            self.g_Na = 4
            self.g_K = 4
        else:
            self.g_L = 8
            self.g_Na = 20
            self.g_K = 10
        
        # Bifurcation occured?
        self.bifur=False
        
        
    def steady_state_IV(self):
        return 1 / (1 + np.exp((self.V_v_half - self.V) / self.V_k))
        
    def steady_state_n(self, V=None):
        if not isinstance(V, type(None)):
            self.V = V
        return 1 / (1 + np.exp((self.n_v_half - self.V) / self.n_k))
    
    def activation_n(self):
        return (self.steady_state_n() - self.n) / self.tau 
    
    def v_nullcline(self, I=0):
        return (I - self.g_L * (self.V - self.E_L) - self.g_Na * self.steady_state_IV() * (self.V - self.E_Na)) / (self.g_K * (self.V - self.E_K))
    
    def bifurcation_diagram(self, V):
        self.V = V
        return self.g_L * (self.V - self.E_L) + self.g_Na * self.steady_state_IV() * (self.V - self.E_Na) + self.g_K * self.steady_state_n() * (self.V - self.E_K) - self.I
    
    def equilibria(self, I=0, init=np.arange(-70, 20,30)):
        self.I = I
        if self.bifur:
            init = [1]
        try:
            x = scipy.optimize.broyden2(self.bifurcation_diagram, init)
            return np.unique(np.round(x, 4))
        except scipy.optimize.nonlin.NoConvergence as e: 
            self.bifur = True
    
    def step(self, I=0):
        # Derivative n
        self.ndot = self.activation_n()
        # Derivative V
        self.Vdot = I - self.g_L * (self.V - self.E_L) - self.g_Na * self.steady_state_IV() * (self.V - self.E_Na) - self.g_K * self.n * (self.V - self.E_K)
        # Update V
        self.V += self.Vdot * self.delta
        # Update n
        self.n += self.ndot * self.delta
        
def trajectory_interactive():
    
    fig, ax = plt.subplots(figsize=(8,8))

    I = 20

    X, Y = np.meshgrid(np.linspace(-80, 20, 15), np.linspace(-.1, 1, 15))
    x = np.linspace(-80, 20, 1000)

    model = NaKModel(X,Y, low_threshold_K=True)
    model.step(I = I)
    U = model.Vdot
    V = model.ndot
    model1 = NaKModel(x, np.zeros(1000), low_threshold_K=True)
    n_nullcline = model1.steady_state_n()
    V_nullcline = model1.v_nullcline(I = I)

    tm = ax.quiver(X, Y, U, V * 200, U)
    ax.set(
        title="Phase Portrait",
        xlabel="membrane voltage, V",
        ylabel="K+ activation variable, n"
    )
    ax.plot(x, n_nullcline, linestyle='dashed', lw=2)
    ax.plot(x, V_nullcline, linestyle='dashed', lw=2)

    xdata, ydata = [], []
    ln, = ax.plot(xdata, ydata, '-', color='lightblue')
    ln1, = ax.plot([], [], 'o', color='lightcoral')

    model = 0
    coords = []

    def animate(frame):
        if model == 0:
            pass
        else:
            xdata.append(model.V)
            ydata.append(model.n)
            model.step(I = I)
            ln.set_data(xdata, ydata)
            ln1.set_data([xdata[-1]], [ydata[-1]])


    def onclick(event):
        V0, n0 = event.xdata, event.ydata
        global model
        model = NaKModel(V0, n0, delta=.01, low_threshold_K=True)

        global coords
        coords.append((V0, n0))

        if len(coords) == 1:
            fig.canvas.mpl_disconnect(cid)

        return coords

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ani = FuncAnimation(fig, animate, frames=200000, repeat=False, interval=20)
    return fig, cid, ani