# Neurodynamics: Script - Bifurcations

### 1. Introduction

I will present the topic bifurcations on two levels. At first I will give a simple mathematical introduction using the logistic map and after that I will explain the relation of bifurcations to Neurodynamics with a two dimensional neuronal model.

The logistic map is an equation that can be interpreted in many different ways. The biologist, Robert May, who first described it for practical purposes used this equation to model the temporal development of a population. Where $x_n$ is the population size at the current time step and $x_{n+1}$ the population size at the next time. This means that you can iteratively apply this computation an arbitrary amount of times to get the population size at any timestep. The system has only one parameter that needs to be set, the growth rate of the population, $r$. However, of course, you also have to make a choice for the initial population size $x_0$. The logistic map is a one dimensional system because it only has one state variable, the population size. A solution to such a system is a trajectory on the phase plane that describes the temporal development of the state variables, given a set of parameter values. 

But what does it mean for a system to undergo a bifurcation? When a system bifurcates the qualitative behavior of its state variables changes. When we want to investigate the qualitative behavior we look at the asymptotic behavior of the trajectories. This means not the first timesteps, but a large number of them. Here you can see a solution to the logistic map that depends on the initial population size and the growth rate.  Notice that it not shows the phase plane, so only the state variables on the axes, instead it depicts the single state variable, the population size, on the y-axis and the time on the x-axis. As you can see here, the population size starts to converge to a specific value after a few timesteps. Those points are called fix points or equilibria, as you learned in the previous video, that attract trajectories from a range of initial conditions. We can also see this property when varying the initial population size. In the range between zero and one all the solutions converge to the same point when the growth rate stays constant. Although the shape of the trajectory at the beginning varies. Now the questions arise, when does a qualitative change of the trajectories occur and how can we determine what counts as a qualitative change? To answer the first question: The qualitative change happens because of the influence of one or more bifurcation parameters, for the logistic map this is the growth rate r. As you can see when we change the growth rate the location of the fix point changes and the trajectory converges to a different value. Although the asymptotic behavior changes, this is not a bifurcation because in the limit the trajectory is topologically equivalent to the previous one. This means that you can transform the "latter part" of the trajectory into the other one by simply shrinking and stretching it. Note that you can apply this kind of reasoning only for trajectories with equal initial population size. There are cases in which the trajectories are not topologically equivalent anymore. When we increase the growth rate further the trajectory starts two show cyclic behavior asymptotically because the population size alternates between two values. The previous trajectory that converged to one value can only be transformed into this one by non-linear operations. This shows us that the system has undergone a bifurcation. As the growth rates increases the logistic map bifurcates many more times.

To illustrate how a system bifurcates one can use a bifurcation diagram. This depicts which values are visited by the system asymptotically depending on the bifurcation parameter. The population size is given on the y-axis and the growth rate on the x-axis. In this plot you can see what we noticed before. In this range the population size just converges to one value and then bifurcates to show this alternating behavior between two values. Then the system bifurcates several times after short successive intervals where the amount of values the system alternates between increases. And finally the system ends up in chaos, but this does not need to interest us for now.

### 2. Bifurcations and Neurodynamics

#### 2.1 $I_{Na,p} + I_K$ - model

Now I would like to explain the relation of bifurcations to neuronal dynamics. To demonstrate this I will use the $I_{Na,p} + I_K$ - model, which is a 2D system that is, unlike the logistic map, described by differential equations. This is a model of a simple single compartment and leaky neuron. The state variables are the membrane voltage and the voltage dependant potassium current activation variable n. The membrane voltage is controlled by three different types of currents. Firstly the leak current, which summarizes the voltage independent currents that flow across the membrane. This means that the permeability of the membrane for these types of currents does not change depending on the voltage. Secondly, the instantaneous sodium current which corresponds to the opening of all voltage dependant sodium channels in an instant. This is of course a simplification, but the error induced by this assumption is not that great compared to neurophysiological recordings in real neurons. The reason for this is that the sodium current has really fast dynamics in most cases. To account for this error we would have to include an additional activation variable just as in the Hodgkin Huxley neuron. However, this would increase the complexity of the model greatly. In contrast to this the third current, the voltage dependant potassium current, is controlled by the activation variable n, which corresponds to the percentage of open potassium channels. In this case the activation variable is more important because the potassium current is relatively slower than the sodium current. The sodium current is an outward current because it decreases the membrane potential and the potassium current is an inward current because it decreases the membrane potential. This model can show many different behaviors depending on the choices you make for the different parameters of the model. We will use parameters values for the neurophysiological properties which have been determined in experiments with real neurons, for example with simple patch clamp experiments. Furthermore we will consider two major variations of this mode: Voltage dependant potassium current with low or high threshold. These properties influence among others whether the neuron is a resonator or an integrator and the type of bifurcation the model undergoes. 

To consider bifurcations we first need to know what the bifurcation parameter of interest is. Here, as in essentially every neuron, this is the injected current, I, that is added to the neuron, for example by another neuron or an experimenter. Of course other parameter changes can also lead to bifurcations. But these are not important for us because we make the assumption that they stay constant in neurons. 

#### 2.2 Recap

There exist in total four different types of bifurcations neurons undergo. I will go through each of these conceptually and talk about some of the relations to the neurophysiological behavior of neurons. At first I will give a short refresher on the types of trajectories we can observe in 2D systems of the form:

$$\dot x = f(x, y)$$

$$\dot y = g(x,y)$$

and show some examples in the $I_{Na,p} + I_K$ - model with high threshold for the potassium current. Where $x$ corresponds to the membrane voltage $V$ and $y$ to the potassium current activation variable $n$. In this plot you can see the phase portrait of the model. On the x-axis is the membrane voltage $V$ and on the $y$-axis the activation variable $n$. In the background the vector field is depicted which gives an idea of how $\dot V$ and $\dot n$ depend on the current state of the system.

There are three different basic types of equilibria, node, saddle and focus. To find these equilibria we have to determine the intersections of the nullclines of the state variables. The nullclines are defined as:

$$f(x,y) = 0$$

$$g(x,y) = 0$$

This means that at any point that satisfies for example the first equation, $x$ does not change because $\dot x = 0$. In this phase portrait you can see the membrane voltage nullcline, the $V$-nullcline, in green and the activation variable nullcline, the $n$-nullcline, in red. Every combination of state variables on the $V$-nullcline results in no change in the membrane voltage. The same holds for the $n$-nullcline. This also shows why we have to search for the intersection of the nullclines to determine the position of the equilibria. By definition equilibria are at points where trajectories that start there are staying there forever. This is only the case when both $\dot x$ and $\dot y$ are zero because that implies that both $x$ and $y$ do not change.

To determine of which type an equilibrium is we have to look at the Eigenvalues of the Jacobian at the position of the equilibrium. I will not explain this here since that is a bit to mathy for this conceptual introduction but you can look that up if you like.

So a node equilibrium can be either stable or unstable, trajectories simply converge to or diverge from it. The saddle is always unstable. Lastly a focus equilibrium can be also either stable or unstable, but the important difference to the node is that trajectories converge or diverge in a cyclic manner.

This leads us to the last important type of asymptotic behavior of trajectories: Limit Cycles. Trajectories that start on a limit cycle form a closed loop. Similarly to equilibria limit cycles either attract or repel trajectories. Here, in the stable case, any trajectory that starts sufficiently near to the limit cycle approaches it and rotates around it forever. In the unstable case the trajectory diverges in a cyclic manner from the limit cycle. This cyclic behavior reminds of the focus equilibrium. And indeed, in limit cycles there is always at least one equilibrium and in most cases this is a focus. For example in the center of stable limit cycles an unstable focus repels the trajectories and from there, they converge to the limit cycle.

Finally, I would like to explain briefly how we can determine whether a bifurcation has occurred in a 2D dynamical system. To do so we look at the difference in the phase portraits between two different choices of the bifurcation parameter. These phase portraits are qualitatively different when they are not topologically equivalent. Just as before this means that you cannot equate the phase portraits by "stretching and shrinking". This happens for example when equilibria change their stability.

#### 2.3 Saddle-node on invariant circle bifurcation

Now I will explain the four types of bifurcations. The neurons in which these bifurcations occur can be classified into four different categories. The properties that differentiate the neurons are their stability, so is the neuron mono- or bistable, and whether they show subthreshold oscillations. The type of bifurcation I will start with is the saddle node bifurcation on invariant circle. This is bifurcation is shown by neurons which are monostable and have no sub-threshold oscillations. We can observe this behavior in the sodium and potassium model with high threshold and relatively slow activation of the voltage gated potassium channels. This neuron is referred to as integrator because it temporally integrates injected currents linearly. 

Here the phase portrait of this model is shown. We can see that there are three equilibria from the intersections of the nullclines. A stable node, a saddle and an unstable focus. When we approximate solutions to this model, we can see by looking at the trajectories that the system always converges to the stable node. It also does not matter whether the initial point is below or above the threshold for the sodium current. When it is above it generates one action potential and converges to the node corresponding to the resting potential. So the system is indeed monostable and also shows no sub threshold oscillations. When we now ramp up the injected current I the node and the saddle equilibrium first approach and then annihilate each other. This means that the two equilibria disappear after the bifurcation. At the point of the bifurcation there is one equilibrium left that is called saddle-node. It has the special property that it attracts trajectories on the node side and repels trajectories on the saddle side. This bifurcation is demonstrated in this animation. On the right the bifurcation diagram is shown. So the V-nullcline shifts upwards because of the injected ramp current. This leads to the annihilation of the two equilibria. After the bifurcation only the unstable focus is left and a stable limit cycle has appeared. However, you cannot see this cycle in the bifurcation diagram. Thats why we will now have a look the next animation. Here we approximate a trajectory while we ramp up the injected current which is shown at the bottom. On the right you can see the membrane voltage plotted against time and on the left you can see the trajectory on the phase plane. So before the bifurcation we can see that the trajectory converges to the node equilibrium that slowly approaches the saddle until it gets annihilated. After the bifurcation the system goes over into periodic spiking which corresponds to tonic firing of action potentials. Lastly I would like to answer the questions why this is called saddle-node bifurcation on invariant circle. The reason for this is that at the point of the bifurcation an invariant circle appears that starts and ends at the saddle-node equilibrium (homoclinic orbit). This circle is called invariant because every solution that starts on this circle stays on it forever. This corresponds to firing of action potentials with zero frequency. After the injected current surpasses the bifurcation point and the saddle-node disappears, the invariant circle becomes a stable limit cycle where the frequency of spikes increases as the injected current increases.

#### 2.4 Saddle-node bifurcation

This leads us directly to the second type of saddle node bifurcation. This bifurcation can be shown in the same model as before, the only difference is that the voltage dependent potassium currents are activated quickly after the threshold is reached. This makes the neuron bistable. As you can see here from the trajectories, there is a stable limit cycle on the right, in addition to the three equilibria which stayed the same. Any trajectory that starts sufficiently near to this cycle converges to it and cycles around it forever. This behavior corresponds to periodic spiking of the neuron. When this neuron bifurcates basically the same event as before happens: Saddle and node approach and annihilate each other as the injected current increases. But here no invariant circle appears and becomes a limit cycle. Instead, the trajectories which would have converged to the node simply jump to the stable limit cycle that prior to the bifurcation coexisted with the resting state. So similar to the previous bifurcation here the neuron also transitions to periodic spiking activity.

#### 2.5 Subcritical Andronov-Hopf bifurcation

The next group of bifurcation types are called Andronov-Hopf bifurcations. These occur in neurons that exhibit damped subthreshold oscillations. These neurons are referred to as resonators because they because the "prefer" a particular frequency for the input. This means that inputs that "resonate" with the neuron lead to faster generation of action potentials. Firstly, I will explain the subcritical Andronov-Hopf bifurcation which neurons undergo that are bistable. To show this bifurcation in the $I_{Na,p} + I_K$ - model we have to adjust it such that the voltage dependant potassium current has a low threshold and a steep activation curve for the instantaneous sodium current. As you can see on the phase portrait, the model has one stable equilibrium at the intersection of the nullclines, a small unstable limit cycle around the focus and one large stable limit cycle. The cycles are illustrated on the bifurcation diagram by depicting only the amplitude, that is the minimum and maximum voltage, of the cycles instead of the whole range of values. In the bifurcation diagram solid lines correspond to the stable equilibrium and cycle and the dashed lines to the unstable. In this model increasing the injected current leads to shrinking of the unstable limit cycle. At the bifurcation point this cycle coalesces with the stable focus. This causes the unstable limit cycle to disappear and the focus to loose its stability. So after the bifurcation the neuron is monostable and exhibits periodic spiking because only the stable limit cycle is left.  

#### 2.6 Supercritical Andronov-Hopf bifurcation

The last type of bifurcation is the supercritical Andronov-Hopf bifurcation. To show the characteristics of this bifurcation we change the voltage dependant sodium current to slower activation dynamics. This model is monostable and has only one stable equilibrium, corresponding to the rest state, and no limit cycles. In this system the equilibrium also looses its stability at the bifurcation point. The difference is that in this case a stable limit cycle appears which amplitude increases as the injected current increases. So the system is still monostable, but transitions to periodic spiking with increasing amplitude.

To conclude I would like to make some final remarks. Firstly a common and important property of neurons is that they are at rest often quite near to the bifurcation point. This gives neurons the ability to quickly transition to and from the spiking state. Secondly, all the bifurcations essentially showed some kind of spiking behavior after the bifurcation had occurred. On first sight this might appear like all these kinds of bifurcations are obsolete. But a more in depth analysis of the intricate neurophysiological and mathematical properties, which I refrained from in large parts of my presentation, reveals really important differences. For example on timing, frequency, amplitude and interactions in neuronal networks these bifurcation types have great influence. 

These simple mathematical descriptions of neuronal dynamics give us the powerful tools to explain many different types of observed neuronal patterns.

### 3. References

1. Izhikevich, Eugene M. *Dynamical systems in neuroscience*. MIT press, 2007.
2. Logistic map. (2001). Wikipedia. Retrieved 20 August, 2020, from https://en.wikipedia.org/wiki/Logistic_map