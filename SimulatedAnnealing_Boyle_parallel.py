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
#import cas9_model.CRISPR_dCas9_binding_curve_Boyle as dCas9Boyle

from time import time
'''
Main function
'''
def sim_anneal_fit(xdata, ydata, yerr, Xstart, lwrbnd, upbnd, model='I_am_using_multi_processing_in_stead',
                   objective_function='chi_squared', on_target_function = 'function_name',
                Tstart=0.1, delta=2.0, tol=1E-3, Tfinal=0.01,adjust_factor=1.1, cooling_rate=0.85, N_int=1000,
                 AR_low=40, AR_high=60, use_multiprocessing=False, nprocs=4, use_relative_steps=True,
                   output_file_results = 'fit_results.txt',
                   output_file_monitor = 'monitor.txt',
                   output_file_init_monitor='init_monitor.txt'):
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


    start_time = time()
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
                   use_relative_steps=use_relative_steps,
                   objective_function=objective_function,
                   on_target_function= on_target_function)

    # Adjust initial temperature
    InitialLoop(SA, X, xdata, ydata, yerr, lwrbnd, upbnd, output_file_init_monitor,start_time)
    print('Initial temp:  ', SA.T)
    # store initial Temperature
    SA.initial_temperature = SA.T


    # Open File for intermediate fit results:
    OutputFitResults = open(output_file_results,'w',1)  #third argument will force the I/O to write into the file every line
    for k in range(len(X)):
        OutputFitResults.write('Parameter ' + str(k+1) + '\t')
    OutputFitResults.write('Potential' + '\t')
    OutputFitResults.write('Equilibruim')
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

            if SA.EQ:
                Eavg /= SA.interval

                # Input this into the check for the stopcondition
                # Call update_temperature() and checks if global stop condition is reached:

                Temperature_Cycle(SA, Eavg)

                # update monitor file:
                write_monitor(SA, output_file_monitor,steps,start_time)

                # Reset the cummalative sum/ average Energy:
                Eavg = 0

                # stop the entire algorithm if condition is met:
                if SA.StopCondition:
                    break
            else:
                write_monitor(SA, output_file_monitor,steps,start_time)  # might want to ommit this call and only write if SA.EQ == True (see call below)

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
        print('Close workers..')
        for i in range(SA.nprocs):
            SA.inQ.put(None)
        for w in SA.processes:
            w.join()
        
    print('Final Temp: ', SA.T)
    print('Final Stepsize: ', SA.step_size)
    print('final solution: ', X)

    #close files:
    OutputFitResults.close()

    return X



'''
Model (Potential)
'''
# Single core model function (for multicore see multiprocessing section)
def chi_squared(xdata, ydata, yerr, params,model):
    # Calculate residuals
    model_result = model(xdata,params)
    residual = ( (model_result-ydata)/yerr )**2
    return residual


def LogLikeLihood(xdata,ydata,yerr, params, model):
    print('param ', params)
    P = model(xdata, params)
    LLike = -np.log(P)
    return LLike


def RelativeError(xdata,ydata,yerr,params,model):
    model_result = model(xdata,params)
    relative_error =  ( (model_result - ydata) / ydata )**2
    return relative_error




def V(SA, xdata,ydata,yerr,params):

    '''
    :param xdata: datapoints
    :param ydata: measured values
    :param yerr: measurement error
    :param params: parameters of model to fit
    :return: Chi^2 value, LogLikeLihood value... the value of the objective function to be minimized
    '''


    # ********** SPECIFIC TO Boyle et al. dataset & position independent model **************
    # on_target_occupancy,_,_ = dCas9Boyle.calc_Boyle(CalcOccupancy=True, CalcOffRate=False, CalcOnRate=False,
    #            parameters=params, mismatch_positions=[],model_id='init_limit_two_drops_fixed_BP')

#    on_target_occupancy, _, _ = SA.on_target_function(CalcOccupancy=True, CalcOffRate=False, CalcOnRate=False,
#                                                      parameters=params, mismatch_positions=[])
    # ***************************************************************************************


    # Multiprocessing
    if SA.MP:
        # split by xdata. Send each entry to an available core
        for i in range(len(xdata)):
            # Added the on_target_occupancy to the job entry. Only in the case of Boyle data is this needed
            InputJob = [params, xdata[i],ydata[i],yerr[i]] # ,on_target_occupancy]
            SA.inQ.put(InputJob)

        # Retreive the results from the results Que and add together to construct Chi-squared
        objective_sum = 0.0
        for i in range(len(xdata)):
            objective_sum += SA.outQ.get()


    # No multiprocessing
    else:
        objective_sum = np.sum(SA.objective_function(xdata,ydata,yerr,params,SA.model))
    return objective_sum


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
                 AR_low, AR_high, use_multiprocessing, nprocs, use_relative_steps, objective_function, on_target_function):
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
        # start the processes (worker function will be continuously running):
        if self.MP:
            self.nprocs = nprocs
            self.inQ = mp.Queue()
            self.outQ = mp.Queue()
            # In this case you provide the objective function and not the model function (so you return ChiSqrd)
            self.objective_function = objective_function
            self.processes = [mp.Process(target=multiprocessing_main_worker,args=(self.inQ,self.outQ,self.objective_function)) for i in range(self.nprocs)]
            for w in self.processes:
                w.start()
            # print self.processes
        else:
            self.nprocs = None
            self.inQ=None
            self.outQ=None
            self.processes = None

            # Objective function is used differently in this case:
            if objective_function == 'chi_squared':
                self.objective_function = chi_squared
            if objective_function == 'relative error':
                self.objective_function = RelativeError
            if objective_function == 'Maximum Likelihood':
                self.objective_function = LogLikeLihood


        # To properly calculate the on-target's values using different parameterizations.
        self.on_target_function = on_target_function

        self.Monitor = {}
        self.InitialMonitor = {}
        self.RelativeSteps = use_relative_steps
        return


def TakeStep(SA, X, lwrbnd, upbnd):
    '''
    This function produces a trial configuration for the continuous variables(slopes)
    :param SA:
    :param X: current solution
    :return: trial solution
    '''
    delta = SA.step_size
    Xtrial = np.zeros(len(X))
    
    print(X)
    print(delta)

    if SA.RelativeSteps:
        X = np.log(X)
        for i in range(len(X)):
            Xtrial[i] = np.random.uniform(np.max([X[i] - delta, np.log(lwrbnd[i])]),
                                          np.min([X[i] + delta, np.log(upbnd[i])]))
    else:
        for i in range(len(X)):
            Xtrial[i] = np.random.uniform(np.max([X[i] - delta, lwrbnd[i]]),
                                          np.min([X[i] + delta, upbnd[i]]))

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
    Xtrial = TakeStep(SA, X,lwrbnd, upbnd)
    
    # print Xtrial
    # print X

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

    # move to next temperature:
    update_temperature(SA)

    # compare relative change in "equillibrium residuals".
    # If the average energy does not change more then the set tolerance between two consequetive temperatures
    # this means you are sufficiently close to the global minimum:
    tolerance_low_enough = abs(SA.average_energy - Eavg)/SA.average_energy < SA.Tolerance

    # If temperature is low enough, stop the optimisation:
    reached_final_temperature = SA.T < SA.final_temperature

    # Bug fix: If temperature is too high, the average energy will not change.
    # So wait until temperature is low enough before considering the tolerance
    # For now: T < 1% T0
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




'''
Initial Temperature 
'''
def InitialLoop(SA, X, xdata, ydata, yerr, lwrbnd, upbnd, initial_monitor_file, start_time):
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
            write_initial_monitor(AR,SA,initial_monitor_file,steps,start_time)
        X = Metropolis(SA, X, xdata, ydata, yerr, lwrbnd, upbnd)
    return



'''
Multiprocessing functions
'''
def multiprocessing_main_worker(InQ, OutQ,calc_objective_function):
    '''
    This function will continuously check for a new job that has been added to the input Que,
    perform the job and append the result to the output Que.

    Technically this is the only function you are sending to all Processes


    ****  It is assumed here that every job has a seperate {x,y} datapoint with the same parameter set ****

    :param InQ: mp.Que() instance
    :param OutQ: mp.Que() instance
    :param calc_objective_function: Function that returns a single term of the Chi-squared
    :return:
    '''
    while True:
        try:
            # Check if there is a new job loaded to the Que (first core that picks it up will do the trick)
            job = InQ.get()
            if job is None:
                break
            # Unpack the job into the correct input arguments
            parameter_values  = job[0]
            xdata = job[1]
            ydata = job[2]
            yerr  = job[3]


            if len(job)>3:
                addidtional_argument = job[4]
                output = calc_objective_function(parameter_values, xdata, ydata, yerr, addidtional_argument)
            else:
                output = calc_objective_function(parameter_values, xdata, ydata)
            # Perform the job:
            OutQ.put(output)
        except Exception(e):
            print("error!", e.message)
            break



'''
Output Files
'''
def write_monitor(SA, output_file_name,steps,start_time):
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
    SA.Monitor['iterations'] = steps
    SA.Monitor['elapsed time']= (time() - start_time)/3600.

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



def write_initial_monitor(AR, SA,output_file_name,steps,start_time):
    '''
    Additional monitor file while Initial loop is running



    makes a file with following information:
    --------------------------------------
    (last recorded) temperature:
    (last recorded) acceptance ratio:
    (last recorded) stepsize:
    '''

    # Since I want to overwrite the content, I re-open the file
    output_file = open(output_file_name, 'w')


    SA.InitialMonitor['(last recorded) Temperature'] = SA.T
    SA.InitialMonitor['(last recorded) Acceptance Ratio'] = AR
    SA.InitialMonitor['(last recorded) stepsize'] = SA.step_size
    SA.InitialMonitor['iterations']=steps
    SA.InitialMonitor['elapsed time']=  (time() - start_time)/3600.
    for key in SA.InitialMonitor:
        output_file.write(str(key) + ':' + str(SA.InitialMonitor[key]) + '\n' )

    output_file.close()
    return
