##############################################################
#  Simulated Annealing for optimisation
#  Misha Klein
#
#  Multiprocessing additions
#  Koen van der Sanden
#
#
#############################################################
import numpy as np
import multiprocessing as mp

'''
Main function
'''
def sim_anneal_fit(xdata, ydata, yerr, Xstart, lwrbnd, upbnd, model,
                Tstart=0.1, delta=2.0, tol=1E-3, Tfinal=0.01,adjust_factor=1.1, cooling_rate=0.85, N_int=1000,
                 AR_low=40, AR_high=60, use_multiprocessing=False, nprocs=4, use_relative_steps=True,
                   output_file_results = 'fit_results.txt',
                   output_file_monitor = 'monitor.txt'):
    '''
    Use Simmulated Annealing to perform Least-Square Fitting

    :param xdata: measured datapoints (independent variable)
    :param ydata: measured datapoints (value)
    :param yerr:  measurement error
    :param Xstart: first guess for parameter values
    :param lwrbnd: lower bound parameters
    :param upbnd:  upper bound parameters. Parameter X[i] will satisfy: lwrbnd <= X[i] <= upbnd[i]
    :param model: the name of the Python function that calculates the model values. This function must be in the form of:
    model(x_value, parameters) (so parameters should be final argument).
    Unpack the values within this funciton to keep this SimmAnneal code general.

    :param output_file_results: Name of outpur file containing the "best fit" parameter set (and intermediates).
    :param output_file_monitor: Name of output file  containing additional info of the SA-optimisation process
    (see the function: "write_monitor()")

    :return: Optimal parameter set X
    Results are also written to files.

    ///////////
    For remaining input arguments (Tstart,delta,tol, etc.) see the 'SimAnneal class'.
    All of these are settings for the SA-algorithm, such as cooling rate, minimum temperature, tolerance etc.
    ///////////

    '''
    # presets
    X = Xstart
    SA = SimAnneal(model=model,
                   Tstart=Tstart,
                   delta=delta,
                   tol=tol,
                   Tfinal=Tfinal,
                   adjust_factor=adjust_factor,
                   cooling_rate=cooling_rate,
                   N_int=N_int,
                   AR_low=AR_low,
                   AR_high=AR_high,
                   use_multiprocessing=use_multiprocessing,
                   nprocs=nprocs,
                   use_relative_steps=use_relative_steps)

    # Adjust initial temperature
    InitialLoop(SA, X, xdata, ydata, yerr, lwrbnd, upbnd)
    print ('Initial temp:  ', SA.T)
    # store initial Temperature
    SA.initial_temperature = SA.T


    # Open File for intermediate fit results:
    OutputFitResults = open(output_file_results,'w',1)  #third argument will force the I/O to write into the file every line
    for k in range(len(X)):
        OutputFitResults.write('Parameter ' + str(k+1) + '\t')
    OutputFitResults.write('Potential' + '\t')
    OutputFitResults.write('Equilibruim' + '\t')
    OutputFitResults.write('\n')



    # Set initial trial:
    X = Xstart
    SA.potential = V(SA, xdata,ydata,yerr,X)
    # Main loop:
    steps = 0
    Eavg = 0
    while True:
        steps += 1
        # if(steps % 1E3 == 0):
        #     print steps

        # I am starting to check If I switch temperature or if I stop simulation
        if SA.EQ:
            # E will represent the average energy at the current temperature
            # during the cycle of SA.interval steps that has just passed.
            Eavg += V(SA, xdata, ydata, yerr, X)
        if (steps % SA.interval == 0):

            # update the intermediate results:
            write_parameters(X, SA,OutputFitResults)
            write_monitor(SA,output_file_monitor)   # might want to ommit this call and only write if SA.EQ == True (see call below)

            if SA.EQ:
                Eavg /= SA.interval

                # Input this into the check for the stopcondition
                # Call update_temperature() and checks if global stop condition is reached:

                Temperature_Cycle(SA, Eavg)

                # update monitor file:
                write_monitor(SA, output_file_monitor)

                # Reset the cummalative sum/ average Energy:
                Eavg = 0

                # stop the entire algorithm if condition is met:
                if SA.StopCondition:
                    break

            # updates stepsize based on Acceptance ratio and checks if you will update
            #  Temperature next time around (updates value "SimAnneal.EQ" to "TRUE")
            # This happens every SA.interval iterations:
            AcceptanceRatio(SA)

            # Every SA.interval steps we will store the results to enable one to 'opt-out' by interupting the code:


        # Accept or reject trial configuration based on Metropolis Monte Carlo.
        # Input: parameters X, output: updates values of parameters X if accepted
        X = Metropolis(SA, X, xdata, ydata, yerr, lwrbnd, upbnd)


    # Close worker processes
    if SA.MP:
        print ('Close workers..')
        SA.processes.close()
        SA.processes.join()

    print ('Final Temp: ', SA.T)
    print ('Final Stepsize: ', SA.step_size)
    print ('final solution: ', X)

    #close files:
    OutputFitResults.close()

    return X

'''
Model (Potential)
'''
# Single core model function (for multicore see multiprocessing section)
def chi_squared(xdata, ydata, yerr, params,model):

    # # Extract parameters
    # A = params[0]
    # nseed = params[1]
    # C = params[2]


    # Calculate residuals
    model_result = model(xdata,params)
    # model_result = A / (1 + np.exp(-(xdata-nseed) * C))
    residual = ( (model_result-ydata)/yerr )**2
    return residual

def V(SA, xdata,ydata,yerr,params):

    '''
    :param xdata: datapoints
    :param ydata: measured values
    :param yerr: measurement error
    :param params: parameters of model to fit
    :return: Chi^2 value. Residual of weighted least squares (chi-squared).
    '''

    # Multiprocessing
    if SA.MP:
        ind = split_data(SA,xdata)
        worker_results = [SA.processes.apply_async(chi_squared , args = (xdata[ind[k]:ind[k+1]],ydata[ind[k]:ind[k+1]],yerr[ind[k]:ind[k+1]],params,SA.model)) for k in range(SA.nprocs)]

        # Retrieve residuals from different processes / cores
        residual_sum = 0
        for w in worker_results:
            residual_sum += np.sum(w.get())

    # No multiprocessing
    else:
        residual_sum = np.sum(chi_squared(xdata,ydata,yerr,params,SA.model))

    return residual_sum

'''
Ancillerary functions
'''
class SimAnneal():
    '''
    stores all global parameters/settings of the Simmulated Annealing problem

    INPUT
    -----
    model : name of Python function used in the form model(x_values,parameters)
    Tstart : Starting temperature (should not matter, is reset during inital loop)
    delta : initial step size for parameter changes
    tol : Stop condition. If relative change in values is less then tolerance, you're done.
    cooling_rate: Exponential cooling is used (T = T0^{cooling_rate})
    T_final: sets an additional stop condition.  If T<T_final, then we will exit the algorithm.
    N_int : Number of steps between every check of the acceptance ratio / equillibrium
    AR_low: 'lower bound ideal acceptance ratio'
    AR_high: 'upper bound ideal acceptance ratio'. Stepsize (and initially temperature)
              are adjusted to keep the instantaneous acceptance ratio between AR_low and AR_high
    adjust_factor: Adjust the stepsize or temperature to have appreaciable acceptance ratio
    nprocs: Sets the number of processes to create for multiprocessing
    use_multiprocessing: If True the program uses multiprocessing and mp_model_func, otherwise it runs on a single core and uses model_func as the model

    FURTHER MONITORS
    ----------------
    self.EQ: Will you move to next temperature? Did you equillibrate at current temperature?
    self.StopcCondition: Global criteria to stop the optimisation procedure.
    self.potential: Variable to store old potential and save on the number of computations
    self.average_energy: Variable to store the average energy at the previous temperature.
    self.processes: Contains the worker processes (pool) to evaluate the potential
    self.MP: Flag to use multiprocesing or not
    self.Monitor: a Python dictionary that stores the information that will get stored into a file
    self.use_relative_steps: Will take the (natural) logarithm of parameters (and stepsize) before constructing trial solutions.
    (Exponentiates again before calulating potentials)
    '''

    def __init__(self, model, Tstart, delta, tol, Tfinal,adjust_factor, cooling_rate, N_int,
                 AR_low, AR_high, use_multiprocessing, nprocs, use_relative_steps):
        self.model = model
        self.T = Tstart
        self.step_size = delta
        self.Tolerance = tol
        self.initial_temperature = Tstart
        self.final_temperature = Tfinal
        self.alpha = adjust_factor  # Factor to adjust stepsize and/or initial temperature
        self.accept = 0
        self.StopCondition = False
        self.EQ = False
        self.upperbnd = AR_high
        self.lwrbnd = AR_low
        self.cooling_rate = cooling_rate
        self.interval = N_int
        self.potential = np.inf
        self.average_energy = np.inf
        self.MP = use_multiprocessing
        self.nprocs = nprocs


        if self.MP:
            self.processes = mp.Pool(nprocs)
        else:
            self.processes = None

        self.Monitor = {}

        self.RelativeSteps = use_relative_steps
        return


def TakeStep(SA,X):
    '''
    This function produces the trial solution to be checked later for acceptance
    :param SA:
    :param X: current solution
    :return: trial solution
    '''

    delta = SA.step_size
    if SA.RelativeSteps:
        X = np.log(X)
        Xtrial = X + np.random.uniform(-delta, delta, size=len(X))
        Xtrial = np.exp(Xtrial)
        X = np.exp(X)
    else:
        Xtrial = X + np.random.uniform(-delta, delta, size=len(X))

    return Xtrial



def Metropolis(SA, X, xdata, ydata, yerr, lwrbnd, upbnd):
    '''
    Metropolis Algorithm to decide if you accept the trial solution.
    Trial solution (Xtrial) is generated with function ('TakeStep()').
    Solution is always rejected if any of its parameter values are outside user defined bounds.
    Only perform Metropolis step if Xtrial is within the bounds given.
    If you do not enter Metropolis, you by definition rejected the trial solution ('tabula rasa' rule)
    :param SA:
    :param X: current solution
    :param xdata: datapoints
    :param ydata: experimental/data values
    :param yerr: error on experimental data
    :param lwrbnd: user defined lower bound for parameter values
    :param upbnd: user defined upper bound for parameter values
    :return: current solution (rejected Xtrial) or updated solution (accepted Xtrial)
    '''
    # T = SA.T
    # delta = SA.step_size
    # if SA.RelativeSteps:
    #     delta = np.log(SA.step_size)
    #     X = np.log(X)
    #     Xtrial = X + np.random.uniform(-delta, delta, size=len(X))
    #     Xtrial = np.exp(Xtrial)
    #     X = np.exp(X)
    # else:
    #     Xtrial = X + np.random.uniform(-delta, delta, size=len(X))
    # # add limits to the parameter values:
    # for i in range(len(Xtrial)):
    #     Xtrial[i] = min(   upbnd[i], max(lwrbnd[i], Xtrial[i])  )


    Xtrial = TakeStep(SA, X)
    while (Xtrial < lwrbnd).any() or (Xtrial > upbnd).any():
        Xtrial = TakeStep(SA,X)


    # Let V({dataset}|{parameterset}) be your residual function.
    # Metropolis:
    T = SA.T
    Vnew = V(SA, xdata, ydata, yerr, Xtrial)
    Vold = SA.potential
    if (np.random.uniform() < np.exp(-(Vnew - Vold) / T)):
        X = Xtrial
        SA.accept += 1
        SA.potential = Vnew
    return X


def AcceptanceRatio(SA):
    AR = (SA.accept / float(SA.interval)) * 100
    if AR > SA.upperbnd:
        SA.step_size *= SA.alpha
    elif AR < SA.lwrbnd:
        SA.step_size /= SA.alpha
    else:
        SA.EQ = True  # <--- the next time around you'll go to TemperatureCycle()
    SA.accept = 0  # reset counter
    return


def Temperature_Cycle(SA, Eavg):
    # compare relative change in "equillibrium residuals".
    # If the average energy does not change more then the set tolerance between two consequetive temperatures
    # this means you are sufficiently close to the global minimum:
    tolerance_low_enough = abs(SA.average_energy - Eavg)/SA.average_energy < SA.Tolerance

    # move to next temperature:
    update_temperature(SA)

    # If temperature is low enough, stop the optimisation:
    reached_final_temperature = SA.T < SA.final_temperature

    # Bug fix: If temperature is too high, the average energy will not change.
    # So wait until temperature is low enough before considering the tolerance
    # For now: T < 10% T0
    temperature_low_enough = SA.T < (0.01 * SA.initial_temperature)

    # check stop condition
    SA.StopCondition = (tolerance_low_enough and temperature_low_enough) or reached_final_temperature

    # done with equillibrium <--> reset
    SA.EQ = False

    # Monitor (I assumed you come to this point at least once, otherwise there is not much to monitor anyways):
    SA.Monitor['reached final temperature'] = reached_final_temperature
    SA.Monitor['tolerance low enough']  = tolerance_low_enough
    SA.Monitor['(last recorded) relative change average energy'] = abs(SA.average_energy - Eavg)/SA.average_energy

    # Update average energy
    SA.average_energy = Eavg
    return

def update_temperature(SA):
    SA.T *= SA.cooling_rate
    return

def InitialLoop(SA, X, xdata, ydata, yerr, lwrbnd, upbnd):
    '''
    Finds starting temperature for SA optimisation by performing some initial iterations until acceptance ratio
    is within acceptable bounds.
    :param SA:
    :param X:
    :param xdata:
    :param ydata:
    :param yerr:
    :param lwrbnd:
    :param upbnd:
    :return:
    '''
    steps = 0
    while True:
        steps +=1
        if (steps % SA.interval == 0):
            AR = (SA.accept / float(SA.interval)) * 100
            if AR > SA.upperbnd:
                SA.T /= SA.alpha
                SA.accept = 0
            elif AR < SA.lwrbnd:
                SA.T *= SA.alpha
                SA.accept = 0
            else:
                SA.accept = 0
                break
        X = Metropolis(SA, X, xdata, ydata, yerr, lwrbnd, upbnd)
    return



'''
Multiprocessing functions
'''
# Gets the indices at which to split the data to distribute among the running processes
def split_data(SA,data):

    length_data = len(data)
    if length_data%SA.nprocs == 0:
        split_list = np.arange(0, length_data, length_data/SA.nprocs)
    else:
        split_list = np.arange(0, length_data-(length_data%SA.nprocs),
                               length_data/SA.nprocs)
        for i in range(length_data%SA.nprocs):
            split_list[i+1:] += 1

    split_list = split_list.tolist()
    split_list.append(None)

    return split_list




'''
Output Files
'''
def write_monitor(SA, output_file_name):
    '''
    makes a file with following information:
    --------------------------------------
    why did the code stop?:   {final temp, manually interupted, ...  }
    (last recorded) temperature:
    (last recorded) stepsize:
    (last recorded) Chi-squared:
    (last recorded) average energy difference:
    '''

    # Since I want to overwrite the content, I re-open the file
    output_file = open(output_file_name, 'w')

    SA.Monitor['(last recorded) Temperature'] = SA.T
    SA.Monitor['(last recorded) stepsize'] = SA.step_size
    SA.Monitor['(last recorded) chi-squared'] = SA.potential
    SA.Monitor['succes'] = SA.StopCondition

    for key in SA.Monitor:
        output_file.write(str(key) + ':' + str(SA.Monitor[key]) + '\n' )

    output_file.close()
    return


def write_parameters(X, SA, output_file):
    '''
    write current parameter set to file:
    param_0||param_1|| ..... || param_N-1 || Potential || Equillibrium
    ------------------------------------------------------------------
        .  ||   .   ||  .    ||  .        ||    .       || TRUE/FALSE
        .  ||   .   ||  .    ||  .        ||    .       || TRUE/FALSE
        .  ||   .   ||  .    ||  .        ||    .       || TRUE/FALSE
        .  ||   .   ||  .    ||  .        ||    .       || TRUE/FALSE

     (delimeter is a tab)

     The last stored set will be the result of the optimasation process
     if the simulated annealing ran until completion.
    '''
    for parameter in X:
        output_file.write(str(parameter) + '\t')
    output_file.write(str(SA.potential) + '\t')
    output_file.write(str(SA.EQ) + '\t')
    output_file.write('\n')
    return
