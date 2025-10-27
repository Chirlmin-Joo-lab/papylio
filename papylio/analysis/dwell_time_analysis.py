import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numbers

import papylio

class ExponentialDistribution:
    """
    A class representing a mixture of exponential distributions.
    """
    def __init__(self, number_of_exponentials, P_bounds=(-1,1), k_bounds=(1e-9,np.inf), truncation=None,
                 sampling_interval=None):
        """
        Initializes the ExponentialDistribution.

        Parameters
        ----------
        number_of_exponentials : int
            Number of exponential components.
        P_bounds : tuple, optional
            Bounds for probability parameters.
        k_bounds : tuple, optional
            Bounds for rate constants.
        truncation : tuple or None, optional
            Range for truncation.
        sampling_interval : float or None, optional
            Interval for sampling.
        """
        self.number_of_exponentials = number_of_exponentials
        self.P_bounds = P_bounds
        self.k_bounds = k_bounds
        # self.bounds = np.array([(0, np.inf)] * (2 * self.number_of_exponentials - 1))
        self.bounds = np.array([P_bounds] * (self.number_of_exponentials - 1) + [k_bounds] * self.number_of_exponentials)
        self.truncation = list(truncation)
        self.sampling_interval = sampling_interval

    def __call__(self, t, *parameters):
        """
        Computes the probability density function (PDF) for given parameters.

        Parameters
        ----------
        t : array-like
            Time values.
        parameters : list
            Model parameters.

        Returns
        -------
        array
            Computed PDF values.
        """
        return self.pdf(t, *parameters)

    @property
    def parameter_names(self):
        """
        Returns the parameter names excluding the last probability term.

        Returns
        -------
        list
            List of parameter names.
        """
        return ([f'P{i}' for i in range(self.number_of_exponentials-1)] +
                [f'k{i}' for i in range(self.number_of_exponentials)])

    @property
    def parameter_names_full(self):
        """
        Returns all parameter names including the last probability term.

        Returns
        -------
        list
            List of all parameter names.
        """
        return ([f'P{i}' for i in range(self.number_of_exponentials)] +
                [f'k{i}' for i in range(self.number_of_exponentials)])

    # def normalize_P(self, P, k):
    #     # P = np.abs(P)
    #     P = np.array(P)
    #     k = np.array(k)
    #     # return P / np.sum(P * (1/k))
    #     return P

    def P_and_k_from_parameters(self, parameters):
        """
        Extracts probabilities and rate constants from parameter list.

        Parameters
        ----------
        parameters : list or array
            Input parameter values.

        Returns
        -------
        tuple
            P (array): Probability values.
            k (array): Rate constants.
        """
        parameters = np.array(parameters)
        P = parameters[0:(self.number_of_exponentials-1)]
        P = np.hstack([P, [1-np.sum(P)]])
        k = parameters[(self.number_of_exponentials - 1):(self.number_of_exponentials*2 - 1)]
        return P, k

    def pdf(self, t, *parameters):
        """
        Computes the probability density function (PDF).

        Parameters
        ----------
        t : array-like
            Time values.
        parameters : list
            Model parameters.

        Returns
        -------
        array
            Computed PDF values.
        """
        # Parameters given as P1, P2, k1, k2, k3
        t = np.array(t)
        pdf = np.zeros_like(t).astype(float)
        P, k = self.P_and_k_from_parameters(parameters)
        for i in range(self.number_of_exponentials):
            pdf += P[i] * k[i] * np.exp(-k[i] * t)

        if self.truncation is not None:
            truncated_probability = (self.cdf_untruncated(self.truncation[1], *parameters) -
                                     self.cdf_untruncated(self.truncation[0], *parameters) + 1e-10)
            pdf = pdf / truncated_probability
            pdf[(t < self.truncation[0]) | (t > self.truncation[1])] = 0

        return pdf

    def cdf_untruncated(self, t, *parameters):
        """
        Computes the cumulative distribution function (CDF) without truncation.


        Parameters
        ----------
        t : array-like
            Time values.
        parameters : list
            Model parameters.

        Returns
        -------
        array
            Computed CDF values.
        """
        t = np.array(t)
        cdf = np.ones_like(t).astype(float)
        P, k = self.P_and_k_from_parameters(parameters)
        for i in range(self.number_of_exponentials):
            cdf -= P[i] * np.exp(-k[i] * t)
        return cdf

    def cdf(self, t, *parameters):
        """
        Computes the cumulative distribution function (CDF), applying truncation if needed.

        Parameters
        ----------
        t : array-like
            Time values.
        parameters : list
            Model parameters.

        Returns
        -------
        array
            Computed CDF values.
        """
        if len(parameters) == self.number_of_exponentials * 2 - 1 + 1:
            self.truncation[0] = parameters[-1]
            parameters = parameters[:-1]
        # elif self.truncation is not None:
        #     truncation = self.truncation
        # else:
        #     truncation = None

        cdf = self.cdf_untruncated(t, *parameters)

        if self.truncation is not None:
            cdf_at_truncation = self.cdf_untruncated(self.truncation, *parameters)
            cdf = (cdf - cdf_at_truncation[0]) / (cdf_at_truncation[1] - cdf_at_truncation[0])
            cdf[(t < self.truncation[0])] = 0
            cdf[(t > self.truncation[1])] = 1

        return cdf

    def pdf_binned(self, t, *parameters):
        """
        Computes the probability density function (PDF) for binned data.

        Parameters
        ----------
        t : array-like
            The time points at which to evaluate the binned PDF.
        *parameters : tuple
            Model parameters used for computing the PDF.

        Returns
        -------
        array-like
            The computed binned PDF values.
        """

        # Parameters given as P1, P2, k1, k2, k3
        # result = np.zeros_like(t).astype(float)
        # P, k = self.P_and_k_from_parameters(parameters)
        # for i in range(self.number_of_exponentials):
        #     result += 2 * np.sinh(k[i] * self.bin_width/2) * P[i] * np.exp(-k[i] * t)
        # result /= self.bin_width

        pdf_binned = self.cdf(t+self.bin_width/2, *parameters) - self.cdf(t-self.bin_width/2, *parameters)
        pdf_binned /= self.bin_width
        return pdf_binned

    def likelihood(self, parameters, t):
        """
        Computes the likelihood function for given parameters.

        Parameters
        ----------
        parameters : list
            Model parameters.
        t : array-like
            Time values.

        Returns
        -------
        float
            Computed likelihood value.
        """
        return np.prod(self.pdf(t, *parameters))

    def loglikelihood(self, parameters, t):
        """
        Computes the log-likelihood function.

        Parameters
        ----------
        parameters : list
            Model parameters.
        t : array-like
            Time values.

        Returns
        -------
        float
            Computed log-likelihood value.
        """
        if len(parameters) == self.number_of_exponentials * 2 - 1 + 1:
            self.truncation[0] = parameters[-1]
            parameters = parameters[:-1]

        if self.truncation is not None:
            t = t[(t > self.truncation[0]) | (t < self.truncation[1])]

        probability_density = self.pdf(t, *parameters) + 1e-10
        # probability_density = self.pdf_binned(t, *parameters) + 1e-10

        loglikelihood = np.sum(np.log(probability_density))

        return loglikelihood

    def negative_loglikelihood(self, parameters, t):
        """
        Computes the negative log-likelihood for given parameters and time data.

        Parameters
        ----------
        parameters : array-like
            Model parameters used for computing the log-likelihood.
        t : array-like
            The time data.

        Returns
        -------
        float
            The negative log-likelihood value.
        """
        return -self.loglikelihood(parameters, t)

    def loglikelihood_binned(self, parameters, bin_centers, counts):
        """
        Computes the log-likelihood for binned data.

        Parameters
        ----------
        parameters : array-like
            Model parameters used for computing the likelihood.
        bin_centers : array-like
            Centers of histogram bins.
        counts : array-like
            Observed counts in each bin.

        Returns
        -------
        float
            The log-likelihood value.
        """
        if self.truncation is not None:
            selection = (bin_centers > self.truncation[0]) | (bin_centers < self.truncation[1])
            bin_centers = bin_centers[selection]
            counts = counts[selection]

        probability_density = self.pdf_binned(bin_centers, *parameters) + 1e-10
        loglikelihood = np.sum(counts * np.log(probability_density * self.bin_width))

        return loglikelihood

    def negative_loglikelihood_binned(self, parameters, bin_centers, counts):
        """
        Computes the negative log-likelihood for binned data.

        Parameters
        ----------
        parameters : array-like
            Model parameters used for computing the likelihood.
        bin_centers : array-like
            Centers of histogram bins.
        counts : array-like
            Observed counts in each bin.

        Returns
        -------
        float
            The negative log-likelihood value.
        """
        return -self.loglikelihood_binned(parameters, bin_centers, counts)

    def mle(self, *args, **kwargs):
        """
        Perform maximum likelihood estimation (MLE) for fitting an exponential distribution.

        This method acts as an alias for `maximum_likelihood_estimation`, forwarding all
        arguments to it.

        Parameters
        ----------
        *args : tuple
           Positional arguments passed to `maximum_likelihood_estimation`.
        **kwargs : dict
           Keyword arguments passed to `maximum_likelihood_estimation`.

        Returns
        -------
        xarray.Dataset
           A dataset containing the fitted parameters and metadata.
        """
        return self.maximum_likelihood_estimation(*args, **kwargs)

    def maximum_likelihood_estimation(self, dwell_times, scipy_optimize_method='minimize', free_truncation_min=False, **kwargs):
        """
        Estimate the best-fit parameters using maximum likelihood estimation (MLE).

        This method applies numerical optimization to maximize the likelihood function
        for the given dwell-time data.

        Parameters
        ----------
        dwell_times : array-like
            Observed dwell times to be used for model fitting.
        scipy_optimize_method : str, optional
            The optimization method from `scipy.optimize` (default: 'minimize').
        free_truncation_min : bool, optional
            If True, an additional truncation parameter is included in the optimization (default: False).
        **kwargs : dict
            Additional arguments passed to the optimization function.

        Returns
        -------
        xarray.Dataset
            A dataset containing the estimated parameters, Bayesian Information Criterion (BIC),
            and optimization metadata.
        """
        scipy_optimize_kwargs = dict(bounds = self.bounds)
        if free_truncation_min:
            scipy_optimize_kwargs['bounds'] = np.concatenate([scipy_optimize_kwargs['bounds'], np.array([[0, dwell_times.min()*2]])])

        if scipy_optimize_method in ['minimize', 'basinhopping', 'dual_annealing', 'differential_evolution']:
            scipy_optimize_kwargs['x0'] = self.parameter_guess(dwell_times)

            if free_truncation_min:
                scipy_optimize_kwargs['x0'] += [self.truncation[0]]

        # def constraint(parameters):
        #     P, k = self.P_and_k_from_parameters(parameters)
        #     return np.sum(P * k)
        # if scipy_optimization_method in ['minimize']:
        #     scipy_optimize_kwargs['constraints'] = scipy.optimize.NonlinearConstraint(constraint, 0, np.inf)
        # elif scipy_optimization_method == 'shgo':
        #     scipy_optimize_kwargs['constraints'] = {'type': 'ineq', 'fun': constraint} #TODO: change in scipy v1.11.0

        scipy_optimize_kwargs.update(kwargs)

        optimal_parameters = getattr(scipy.optimize, scipy_optimize_method)(self.negative_loglikelihood,
                                     args = (dwell_times,), **scipy_optimize_kwargs).x
        BIC = self.BIC(dwell_times, optimal_parameters)

        dwell_analysis = self.parameters_to_dataset(optimal_parameters, BIC=BIC)
        dwell_analysis.attrs['fit_method'] =  'maximum_likelihood_estimation'
        dwell_analysis.attrs['scipy_optimize_method'] = scipy_optimize_method

        for key, item in scipy_optimize_kwargs.items():
            if isinstance(item, np.ndarray):
                item = item.tolist()
            elif isinstance(item, bool):
                item = str(item)
            if key == 'bounds':
                dwell_analysis.attrs['scipy_optimize_bounds_min'] = [b[0] for b in item]
                dwell_analysis.attrs['scipy_optimize_bounds_max'] = [b[1] for b in item]
            else:
                dwell_analysis.attrs['scipy_optimize_'+key] = item
        # dwell_analysis.attrs['scipy_optimize_kwargs'] = scipy_optimize_kwargs

        return dwell_analysis

    def hist_fit(self, *args, **kwargs):
        """
        Fit an exponential distribution to binned dwell-time data.

        This method acts as an alias for `histogram_fit`, forwarding all arguments to it.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to `histogram_fit`.
        **kwargs : dict
            Keyword arguments passed to `histogram_fit`.

        Returns
        -------
        xarray.Dataset
            A dataset containing the fitted parameters and metadata.
        """
        return self.histogram_fit(*args, **kwargs)

    def histogram_fit(self, dwell_times, bins='auto_discrete', free_truncation_min=True, remove_first_bins=0, **kwargs):
        """
        Fit an exponential distribution to a histogram of dwell-time data.

        The method constructs a histogram of the dwell times and fits a probability
        density function to the binned data.

        Parameters
        ----------
        dwell_times : array-like
            Observed dwell times used for model fitting.
        bins : int, str, or array-like, optional
            Method for binning the histogram. If 'auto_discrete', an automatic binning
            strategy is applied (default: 'auto_discrete').
        free_truncation_min : bool, optional
            If True, an additional truncation parameter is included in the fitting process (default: True).
        remove_first_bins : int, optional
            Number of initial bins to exclude from the fitting (default: 0).
        **kwargs : dict
            Additional arguments passed to `scipy.optimize.curve_fit`.

        Returns
        -------
        xarray.Dataset
            A dataset containing the estimated parameters, their uncertainties, Bayesian
            Information Criterion (BIC), and metadata about the fitting procedure.
        """

        if bins == 'auto_discrete':
            bin_edges = auto_bin_size_for_discrete_data(dwell_times)
        else:
            bin_edges = np.histogram_bin_edges(dwell_times, bins=bins)

        bin_edges = bin_edges[remove_first_bins:]

        self.bin_width = bin_edges[1] - bin_edges[0]
        weights = (1 / len(dwell_times) / self.bin_width,) * len(dwell_times)  # Assuming evenly spaced bins
        counts, bin_edges = np.histogram(dwell_times, bins=bin_edges, density=False, weights=weights)

        scipy_curve_fit_kwargs = dict(p0 = self.parameter_guess(dwell_times),
                                      bounds = self.bounds.T, absolute_sigma=True)
        scipy_curve_fit_kwargs.update(kwargs)

        if free_truncation_min:
            scipy_curve_fit_kwargs['p0'] += [self.truncation[0]]
            scipy_curve_fit_kwargs['bounds'] = np.concatenate([scipy_curve_fit_kwargs['bounds'].T, np.array([[0, dwell_times.min()*2]])]).T

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        optimal_parameters, parameter_covariances = scipy.optimize.curve_fit(self.pdf_binned, bin_centers, counts,
                                                                            **scipy_curve_fit_kwargs)
        parameter_errors = np.sqrt(np.diag(parameter_covariances))
        BIC = self.BIC_histogram(bin_centers, counts*len(dwell_times)*self.bin_width, optimal_parameters)

        dwell_analysis = self.parameters_to_dataset(optimal_parameters, parameter_errors=parameter_errors, BIC=BIC)
        dwell_analysis.attrs['fit_method'] =  'histogram_fit'
        dwell_analysis.attrs['bins'] = bins
        dwell_analysis.attrs['remove_first_bins'] = remove_first_bins

        for key, item in scipy_curve_fit_kwargs.items():
            if isinstance(item, np.ndarray):
                item = item.tolist()
            elif isinstance(item, bool):
                item = str(item)
            if key == 'bounds':
                dwell_analysis.attrs['scipy_curve_fit_bounds_min'] = item[0]
                dwell_analysis.attrs['scipy_curve_fit_bounds_max'] = item[1]
            else:
                dwell_analysis.attrs['scipy_curve_fit_'+key] = item

        return dwell_analysis
        #
        # fig, ax = plt.subplots()
        # ax.hist(t, bins=bins, density=True)
        # t_fit = np.linspace(0, bin_centers.max(), 1000)
        # t_hist = bin_centers
        # y_fit = self.pdf_binned(t_fit, *optimal_parameters)
        # y_hist = counts
        # ax.plot(t_fit, y_fit)
        # ax.plot(t_hist, y_hist)

    def cdf_fit(self, dwell_times, free_truncation_min=True, **kwargs):
        """
        Fit an exponential distribution to the empirical cumulative distribution function (CDF).

        The method uses curve fitting to match the observed CDF of the dwell times
        to the theoretical CDF of an exponential distribution.

        Parameters
        ----------
        dwell_times : array-like
            The dwell times used for model fitting.
        free_truncation_min : bool, optional
            If True, an additional truncation parameter is included in the fitting process (default: True).
        **kwargs : dict
            Additional arguments passed to `scipy.optimize.curve_fit`.

        Returns
        -------
        xarray.Dataset
            A dataset containing the optimal parameters, parameter uncertainties, Bayesian
            Information Criterion (BIC), and fitting metadata.
        """
        t, ecdf = empirical_cdf(dwell_times, self.sampling_interval)

        scipy_curve_fit_kwargs = dict(p0 = self.parameter_guess(dwell_times),
                                      bounds = self.bounds.T, absolute_sigma=True)
        if free_truncation_min:
            scipy_curve_fit_kwargs['p0'] += [self.truncation[0]]
            scipy_curve_fit_kwargs['bounds'] = np.concatenate([scipy_curve_fit_kwargs['bounds'].T, np.array([[0, dwell_times.min()*2]])]).T

        scipy_curve_fit_kwargs.update(kwargs)

        optimal_parameters, parameter_covariances = scipy.optimize.curve_fit(self.cdf, t, ecdf,
                                                                             **scipy_curve_fit_kwargs)
        parameter_errors = np.sqrt(np.diag(parameter_covariances))

        BIC = self.BIC(dwell_times, optimal_parameters) #TODO: change or check BIC calculation

        dwell_analysis = self.parameters_to_dataset(optimal_parameters, parameter_errors=parameter_errors, BIC=BIC)
        dwell_analysis.attrs['fit_method'] =  'cdf_fit'

        for key, item in scipy_curve_fit_kwargs.items():
            if isinstance(item, np.ndarray):
                item = item.tolist()
            elif isinstance(item, bool):
                item = str(item)
            if key == 'bounds':
                dwell_analysis.attrs['scipy_curve_fit_bounds_min'] = item[0]
                dwell_analysis.attrs['scipy_curve_fit_bounds_max'] = item[1]
            else:
                dwell_analysis.attrs['scipy_curve_fit_' + key] = item

        return dwell_analysis

    def parameter_guess(self, dwell_times):
        """
        Generate an initial guess for the parameters based on the dwell times.

        Parameters
        ----------
        dwell_times : array-like
            The dwell times data to be used for generating the parameter guess.

        Returns
        -------
        parameters : list
            A list of initial guesses for the parameters.
        """
        guess_P = [1 / (i + 1) for i in range(1, self.number_of_exponentials)]
        guess_k = [1 / (dwell_times.mean() * (i + 1)) for i in range(self.number_of_exponentials)]
        parameters = guess_P + guess_k
        return parameters

    def parameters_full(self, parameters):
        """
        Modify the parameters by ensuring that the sum of the first `number_of_exponentials - 1` parameters
        is less than or equal to 1 and adjusting the last parameter accordingly.

        Parameters
        ----------
        parameters : list
            A list of parameters where the first `number_of_exponentials - 1` are P values.

        Returns
        -------
        parameters : list
            The modified list of parameters with the adjusted last parameter.
        """
        parameters = list(parameters)
        P = parameters[0:self.number_of_exponentials-1]
        parameters.insert(self.number_of_exponentials - 1, 1 - sum(P))
        return parameters

    def BIC(self, dwell_times, optimal_parameters):
        """
        Compute the Bayesian Information Criterion (BIC) for the given dwell times and optimal parameters.

        Parameters
        ----------
        dwell_times : array-like
            The dwell times data used to calculate the BIC.
        optimal_parameters : array-like
            The optimal parameters obtained from fitting the model.

        Returns
        -------
        BIC : float
            The Bayesian Information Criterion value.
        """
        return len(optimal_parameters) * np.log(len(dwell_times)) - 2 * self.loglikelihood(optimal_parameters, dwell_times)

    def BIC_histogram(self, bin_centers, counts, optimal_parameters):
        """
        Compute the Bayesian Information Criterion (BIC) for the binned data using the optimal parameters.

        Parameters
        ----------
        bin_centers : array-like
            The centers of the histogram bins.
        counts : array-like
            The counts corresponding to each histogram bin.
        optimal_parameters : array-like
            The optimal parameters obtained from fitting the model.

        Returns
        -------
        BIC : float
            The Bayesian Information Criterion value for the binned data.
        """
        k = len(optimal_parameters)
        N = np.sum(counts)
        BIC = k * np.log(N) - 2 * self.loglikelihood_binned(optimal_parameters, bin_centers, counts)
        return BIC

    # def parameters_to_dataframe(self, parameters, bic=True, dwell_times=None):
    #     dwell_analysis = {}
    #     dwell_analysis['Exponential'] = [self.number_of_exponentials] * self.number_of_exponentials #+
    #     dwell_analysis['Component'] = np.arange(self.number_of_exponentials)+1# [''] * (e-1)
    #     dwell_analysis['P'] = np.hstack([parameters[0:self.number_of_exponentials-1] , np.nan])
    #     dwell_analysis['P'][-1] = 1-np.nansum(dwell_analysis['P'])
    #     dwell_analysis['k'] = parameters[self.number_of_exponentials - 1:]
    #     # if parameter_errors is not None:
    #     #     dwell_analysis['P error'] = np.hstack([parameter_errors[0:e-1] ,np.nan])
    #     #     dwell_analysis['k error'] = parameter_errors[e - 1:]
    #     if bic and dwell_times is not None:
    #         dwell_analysis['BIC']  = [self.BIC(dwell_times, *parameters)] + [np.nan] * (self.number_of_exponentials-1)
    #
    #     return pd.DataFrame(dwell_analysis).set_index('Exponential')
    #
    # def dataframe_to_parameters(self, dataframe):
    #     parameters = dataframe.loc[self.number_of_exponentials][['P', 'k']].values.T.flatten()
    #     # return parameters[~np.isnan(parameters)]
    #     parameters = list(parameters)
    #     parameters.pop(self.number_of_exponentials-1)
    #     return parameters

    def parameters_to_dataset(self, parameters, parameter_errors=None, BIC=None):
        """
        Convert model parameters and associated information into an xarray dataset.

        Parameters
        ----------
        parameters : list
            The model parameters (P and k values) to be included in the dataset.
        parameter_errors : list, optional
            The errors for the model parameters, used if available to include in the dataset.
        BIC : float, optional
            The Bayesian Information Criterion value, used if available to include in the dataset.

        Returns
        -------
        dwell_analysis : xarray.Dataset
            The dataset containing the parameters, errors, BIC value, and other associated metadata.
        """
        # coords = dict(parameter=pd.MultiIndex.from_product((['P','k'], np.arange(self.number_of_exponentials)), names=['name', 'component']))
        # coords = dict(parameter=('parameter', np.repeat(['P','k'],2)),
        #               component=('parameter', list(range(self.number_of_exponentials))*2))
        # parameters_da = xr.DataArray(self.parameters_full(parameters), dims=('parameter'),
        #                              coords=coords,
        #                              name='parameters').expand_dims('fit')
        #
        # bic = xr.DataArray([self.BIC(dwell_times, *parameters)], dims=('fit'))
        # dwell_analysis = xr.Dataset(dict(parameter=parameters, bic=bic))
        data_vars = {} # dict(P=P, k=k, BIC=BIC, fit_function=fit_function, fit_method=fit_method, number_of_components=number_of_components)
        parameters_full = self.parameters_full(parameters)
        P = parameters_full[0:self.number_of_exponentials]
        k = parameters_full[self.number_of_exponentials:self.number_of_exponentials*2]

        data_vars['P'] = xr.DataArray(P, dims=('component')).expand_dims('fit')
        data_vars['P'].attrs['units'] = ''

        data_vars['k'] = xr.DataArray(k, dims=('component')).expand_dims('fit')
        data_vars['k'].attrs['units'] = 's⁻¹'

        if len(parameters_full) > 2 * self.number_of_exponentials:
            truncation = parameters_full[self.number_of_exponentials*2:]
            data_vars['truncation_min'] = xr.DataArray(truncation[0]).expand_dims('fit')
            data_vars['truncation_min'].attrs['units'] = 's'

        if parameter_errors is not None:
            P_error = self.parameters_full(parameter_errors)[0:self.number_of_exponentials]
            P_error[-1] = np.nan
            k_error = self.parameters_full(parameter_errors)[self.number_of_exponentials:self.number_of_exponentials*2]

            data_vars['P_error'] = xr.DataArray(P_error, dims=('component')).expand_dims('fit')
            data_vars['P_error'].attrs['units'] = ''

            data_vars['k_error'] = xr.DataArray(k_error, dims=('component')).expand_dims('fit')
            data_vars['k_error'].attrs['units'] = 's⁻¹'

            if len(parameters_full) > 2 * self.number_of_exponentials:
                truncation_errors = self.parameters_full(parameter_errors)[self.number_of_exponentials*2:]
                data_vars['truncation_min_error'] = xr.DataArray(truncation_errors[0]).expand_dims('fit')
                data_vars['truncation_min_error'].attrs['units'] = 's'

        if BIC is not None:
            data_vars['BIC'] = xr.DataArray([BIC], dims=('fit'))
            data_vars['BIC'].attrs['units'] = ''

        data_vars['number_of_components'] = xr.DataArray([self.number_of_exponentials], dims='fit')

        coords = dict(component=np.arange(self.number_of_exponentials))
        dwell_analysis = xr.Dataset(data_vars=data_vars, coords=coords)
        dwell_analysis.attrs['version'] = papylio.__version__
        dwell_analysis.attrs['fit_function'] = 'exponential'
        dwell_analysis.attrs['truncation'] = self.truncation
        dwell_analysis.attrs['P_bounds'] = self.P_bounds
        dwell_analysis.attrs['k_bounds'] = self.k_bounds
        return dwell_analysis

    def dataset_to_parameters(self, dataset):
        """
        Extract the model parameters from an xarray dataset.

        Parameters
        ----------
        dataset : xarray.Dataset
            The dataset containing the model parameters (P and k values).

        Returns
        -------
        parameters : list
            The extracted model parameters, excluding the last parameter (P value).
        """
        # dataset = dataset.sel(fit=number_of_components == self.number_of_exponentials) # Could be used to assure the correct dataset is passed.
        parameters = dataset[['P', 'k']].to_array().values.flatten()
        parameters = parameters[~np.isnan(parameters)]
        parameters = list(parameters)
        parameters.pop(self.number_of_exponentials-1)
        if 'truncation_min' in dataset.data_vars.keys():
            parameters += [dataset['truncation_min'].item()]
        return parameters

def auto_bin_size_for_discrete_data(dwell_times, sampling_interval=None):
    """
    Calculate the optimal bin edges for a histogram of discrete data using the Freedman-Diaconis rule.

    Parameters
    ----------
    dwell_times : array-like
        The input dwell times, which are sorted and used to calculate the optimal bin edges.

    Returns
    -------
    bin_edges : numpy.ndarray
        The calculated bin edges for the histogram.
    """
    # dwell_times.sort()
    # d = np.diff(dwell_times)
    # bin_width_min = d[d > 0][1]
    if sampling_interval is None:
        sampling_interval = dwell_times.min()

    Q1 = np.percentile(dwell_times, 25)
    Q3 = np.percentile(dwell_times, 75)
    IQR = Q3 - Q1
    bin_width_fd = 2 * IQR / len(dwell_times) ** (1 / 3)

    bin_width = np.ceil(bin_width_fd / sampling_interval) * sampling_interval

    # plot_range = (bin_width / 2, np.percentile(dwell_times, 99))
    # plot_range = (np.min(dwell_times) / 2, np.max(dwell_times))
    plot_range = (np.min(dwell_times) / 2, np.max(dwell_times))

    bin_edges = np.arange(plot_range[0], plot_range[1], bin_width)

    return bin_edges

def plot_dwell_time_histogram(dwell_times, bins='auto_discrete', range=None, sampling_interval=None, ax=None, **hist_kwargs):
    """
    Plot a histogram of dwell times with automatic bin sizing or specified bin edges.

    Parameters
    ----------
    dwell_times : array-like
        The input dwell times to be used for plotting the histogram.
    bins : str or int or sequence, optional
        The binning method, either 'auto_discrete' for automatic bin sizing or a specific bin configuration.
    range : tuple, optional
        The range of values to be used for the histogram.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the histogram. If None, a new figure and axes will be created.
    **hist_kwargs : keyword arguments
        Additional arguments to pass to `matplotlib.pyplot.hist`.

    Returns
    -------
    counts : numpy.ndarray
        The counts for each bin in the histogram.
    bin_centers : numpy.ndarray
        The center positions of each bin.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if bins == 'auto_discrete':
        bins = auto_bin_size_for_discrete_data(dwell_times, sampling_interval)
        bins = bins[(bins>range[0])&(bins<range[1])]
    else:
        bins = np.histogram_bin_edges(dwell_times, bins=bins, range=range)

    bin_width = bins[1] - bins[0]
    weights = (1/len(dwell_times)/ bin_width,) * len(dwell_times) # Assuming evenly spaced bins
    counts, bin_edges, _ = ax.hist(dwell_times, bins=bins, range=range, weights=weights, density=False, **hist_kwargs)
    # relative_counts = counts/(len(dwell_times) * np.diff(bin_edges))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.set_ylim(0,counts.max()*1.03)
    return counts, bin_centers

def empirical_cdf(dwell_times, sampling_interval):
    """
    Compute the empirical cumulative distribution function (ECDF) of dwell times.

    Parameters
    ----------
    dwell_times : array-like
        The input dwell times for which the ECDF is calculated.
    sampling_interval : float
        The sampling interval used to adjust the ECDF calculation.

    Returns
    -------
    t : numpy.ndarray
        The time points at which the ECDF is evaluated.
    empirical_cdf : numpy.ndarray
        The values of the empirical cumulative distribution function.
    """
    # t = np.hstack([0, np.sort(dwell_times)])
    # empirical_cdf = np.arange(0, len(t)) / (len(t) - 1)

    t, counts  = np.unique(dwell_times, return_counts=True)
    t = t + sampling_interval / 2
    empirical_cdf = counts.cumsum() / counts.sum()
    # t = np.hstack([[0,], t])
    # empirical_cdf = np.hstack([[0, ], empirical_cdf])

    return t, empirical_cdf

def plot_empirical_cdf(dwell_times, sampling_interval, ax=None, **plot_kwargs):
    """
    Plot the empirical cumulative distribution function (ECDF) of dwell times.

    Parameters
    ----------
    dwell_times : array-like
        The input dwell times to be used for the ECDF plot.
    sampling_interval : float
        The sampling interval used to adjust the ECDF calculation.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the ECDF. If None, a new figure and axes will be created.
    **plot_kwargs : keyword arguments
        Additional arguments to pass to `matplotlib.pyplot.plot`.

    Returns
    -------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()

    t, ecdf = empirical_cdf(dwell_times, sampling_interval)

    kwargs = {'marker':'.', 'color': 'k', 'ls':'None'}
    kwargs.update(plot_kwargs)
    ax.plot(t, ecdf, **kwargs, zorder=1000)

    # ax.fill_between(t, ecdf, **kwargs)

def fit_dwell_times(dwell_times, method='maximum_likelihood_estimation', number_of_exponentials=[1,2],
                    P_bounds=(-1,1), k_bounds=(0,np.inf), sampling_interval=None, truncation=None, fit_dwell_times_kwargs={}):
    """
    Fit dwell times using a specified fitting method.

    Parameters
    ----------
    dwell_times : array-like
        The input dwell times to be fitted.
    method : str, optional
        The fitting method to use (e.g., 'maximum_likelihood_estimation').
    number_of_exponentials : list of int, optional
        The number of exponentials to fit.
    P_bounds : tuple, optional
        The bounds for the P parameters.
    k_bounds : tuple, optional
        The bounds for the k parameters.
    sampling_interval : float, optional
        The sampling interval used for analysis.
    truncation : tuple, optional
        The truncation limits for the fitting.
    fit_dwell_times_kwargs : dict, optional
        Additional arguments to pass to the fitting method.

    Returns
    -------
    dwell_analysis : xarray.Dataset
        The dataset containing the fitted parameters and analysis results.
    """
    if isinstance(number_of_exponentials, numbers.Number):
        number_of_exponentials = [number_of_exponentials]

    if sampling_interval is None:
        sampling_interval = np.min(dwell_times)

    if truncation is None:
        truncation = (sampling_interval / 2, np.inf)

    dwell_analysis = []
    for i in number_of_exponentials:
        distribution = ExponentialDistribution(i, P_bounds, k_bounds, truncation=truncation,
                                               sampling_interval=sampling_interval)
        # self.bounds[self.bounds[:, 1] > bounds_max, 1] = bounds_max
        dwell_analysis_i = getattr(distribution, method)(dwell_times, **fit_dwell_times_kwargs)
        # dwell_analysis.append(distribution.parameters_to_dataframe(optimal_parameters, BIC=True, dwell_times=dwell_times))
        dwell_analysis.append(dwell_analysis_i)

    # dwell_analysis[e] = {'parameters': optimal_parameters,
    #                   'parameter_errors': parameter_errors,
    #                   'BIC': distribution.BIC(dwell_times, *optimal_parameters) }

    # return pd.concat(dwell_analysis)
    dwell_analysis = xr.concat(dwell_analysis, dim='fit', combine_attrs="drop_conflicts")
    dwell_analysis.attrs['sampling_interval'] = sampling_interval
    return dwell_analysis

def plot_dwell_analysis_state(dwell_analysis, dwell_times, plot_type='pdf_binned', plot_range=None, bins='auto_discrete', log=False, ax=None):
    """
    Plot the results of a dwell time analysis, including the model fit and empirical data.

    Parameters
    ----------
    dwell_analysis : xarray.Dataset
        The dataset containing the dwell time analysis results, including fitted parameters and BIC values.
    dwell_times : array-like
        The input dwell times used for the analysis.
    plot_type : str, optional
        The type of plot to generate, e.g., 'pdf_binned', 'cdf'.
    plot_range : tuple, optional
        The range of values to plot.
    bins : str or int or sequence, optional
        The binning method for the histogram plot.
    log : bool, optional
        If True, the y-axis is plotted on a logarithmic scale.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the results. If None, a new figure and axes will be created.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if plot_range is None:
        plot_range = (0, np.max(dwell_times)) #np.quantile(dwell_times, 0.99))

    sampling_interval = dwell_analysis.sampling_interval

    if plot_type in ['pdf', 'pdf_binned']:
        counts, bin_centers = plot_dwell_time_histogram(dwell_times, bins=bins, range=plot_range,
                                                        sampling_interval=sampling_interval, log=log, ax=ax, color='gainsboro')
        bin_width = bin_centers[1] - bin_centers[0]
    elif plot_type == 'cdf':
        plot_empirical_cdf(dwell_times, sampling_interval, ax=ax)
        ax.set_xlim(*plot_range)
        bin_width = None
        if log:
            ax.set_yscale('log')

    dwell_analysis_formatted = dwell_analysis[['P','k','BIC','number_of_components']].to_dataframe().dropna(subset=['P','k']) #dwell_analysis.copy()
    dwell_analysis_formatted['P'] = [f'{float(x):.2f}' if pd.notna(x) else '' for x in dwell_analysis_formatted['P']]
    dwell_analysis_formatted['k'] = [f'{float(x):.4f} /s' if pd.notna(x) else '' for x in dwell_analysis_formatted['k']]
    dwell_analysis_formatted['BIC'] = [f'{float(x):.0f}' if pd.notna(x) else '' for x in dwell_analysis_formatted['BIC']]
    dwell_analysis_formatted.loc[dwell_analysis_formatted.index.get_level_values('component') > 0, 'BIC'] = ''

    # dwell_analysis_formatted = dwell_analysis_formatted.reset_index()
    # exponentials = dwell_analysis_formatted['Exponential'].values
    # dwell_analysis_formatted.loc[
    #     np.hstack([np.where(exponentials == e)[0][1:] for e in np.unique(exponentials)]), 'Exponential'] = ''
    # dwell_analysis_formatted['Exponential'] = dwell_analysis_formatted['Exponential'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                                                ['Single', 'Double', 'Triple', 'Quadruple', 'Quintuple',
    #                                                 'Sextuple', 'Septuple', 'Octuple', 'Nonuple',
    #                                                 'Decuple'])

    dwell_analysis_formatted = dwell_analysis_formatted.reset_index()
    number_of_components = dwell_analysis_formatted['number_of_components'].values
    dwell_analysis_formatted['Exponential'] = dwell_analysis_formatted['number_of_components'].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                   ['Single', 'Double', 'Triple', 'Quadruple', 'Quintuple',
                                                    'Sextuple', 'Septuple', 'Octuple', 'Nonuple',
                                                    'Decuple'])
    dwell_analysis_formatted.loc[dwell_analysis_formatted.component > 0, 'Exponential'] = ''
    dwell_analysis_formatted = dwell_analysis_formatted.set_index('number_of_components')[['Exponential','P','k','BIC']]

    labels = dwell_analysis_formatted.to_string(header=True, index=False, justify='left').split('\n')
    header = labels.pop(0)
    labels = np.array(labels)

    ax.plot([], [], color='k', linestyle=None, lw=0, marker=None, label=header)

    # for i, number_of_exponentials in enumerate(dwell_analysis.index.unique().values):
    #     distribution = ExponentialDistribution(number_of_exponentials)
    #
    #     label = '\n'.join(labels[dwell_analysis.index == number_of_exponentials])
    #
    #     ax.plot(t, distribution.pdf(t, *distribution.dataframe_to_parameters(dwell_analysis.loc[[number_of_exponentials]])),
    #             label=label)

    for i, number_of_exponentials in enumerate(np.unique(number_of_components)):
        if 'truncation_min' in dwell_analysis.data_vars.keys():
            truncation = [dwell_analysis.truncation_min[i].item(), np.inf]
        else:
            truncation = dwell_analysis.truncation

        distribution = ExponentialDistribution(number_of_exponentials, truncation=truncation,
                                               sampling_interval=sampling_interval)
        distribution.bin_width = bin_width
        label = '\n'.join(labels[dwell_analysis_formatted.reset_index().number_of_components == number_of_exponentials])

        parameters = distribution.dataset_to_parameters(dwell_analysis.sel(fit=dwell_analysis.number_of_components==number_of_exponentials))

        if plot_type == 'pdf_binned':
            t = np.vstack([bin_centers - bin_width / 2, bin_centers + bin_width / 2]).T.flatten()
            y = getattr(distribution, plot_type)(bin_centers, *parameters)
            y = np.repeat(y, 2)
        else:
            t = np.linspace(plot_range[0], plot_range[1], 1000)
            y = getattr(distribution, plot_type)(t, *parameters)

        ax.plot(t, y, label=label)

    legend = ax.legend(prop={"family": "monospace"}, labelcolor='linecolor', frameon=False)
    ax.set_xlabel('Dwell time (s)')
    if plot_type == 'pdf':
        ax.set_ylabel('Normalized counts')
    elif plot_type == 'cdf':
        ax.set_ylabel('Normalized cumulative counts')

    return ax.figure, ax

def analyze_dwells(dwells, method='maximum_likelihood_estimation', number_of_exponentials=[1,2,3], state_names=None,
                   P_bounds=(-1,1), k_bounds=(1e-9,np.inf), sampling_interval=None, truncation=None, fit_dwell_times_kwargs={}):
    """
    Analyze dwell times for different states using a fitting method.

    This function fits the dwell times for each state, allowing for the number of exponentials to be different for each state. The fitting is done using a specified method, such as maximum likelihood estimation. The resulting fit parameters are returned in an `xarray` dataset.

    Parameters:
    ----------
    dwells : xarray.DataArray
        An xarray DataArray containing the dwell times and states to be analyzed.
    method : str, optional
        The method used for fitting the dwell times (default is 'maximum_likelihood_estimation').
    number_of_exponentials : list of int or dict, optional
        A list of integers or a dictionary specifying the number of exponentials to fit for each state. If a dictionary is provided, it maps each state to a specific number of exponentials (default is [1, 2, 3]).
    state_names : dict, optional
        A dictionary mapping state indices to state names (default is None).
    P_bounds : tuple, optional
        The bounds for the parameter P (default is (-1, 1)).
    k_bounds : tuple, optional
        The bounds for the parameter k (default is (1e-9, np.inf)).
    sampling_interval : float, optional
        The sampling interval for the dwell times (default is None, which uses the minimum dwell time).
    truncation : tuple, optional
        A tuple specifying the truncation limits (default is None, which applies no truncation).
    fit_dwell_times_kwargs : dict, optional
        Additional keyword arguments to be passed to the `fit_dwell_times` function.

    Returns:
    -------
    xarray.Dataset
        An `xarray` dataset containing the fitted parameters for each state, including the number of components, P, k, and BIC values.
    """

    # number_of_exponentials can be given per state as {0: [1,2,3], 1: [1,2]}
    if state_names is None:
        states = np.unique(dwells.state)
        states = states[states >= 0]
        # state_names = {state: '' for state in states}
    else:
        states = np.array(list(state_names.keys()))

    if not isinstance(number_of_exponentials, dict):
        number_of_exponentials = {state: number_of_exponentials for state in states}

    # number_of_exponentials_max = np.concatenate(list(number_of_exponentials.values())).max()
    #
    # # fit_parameters = list(inspect.signature(fit_function).parameters)[1:]
    # fit_parameters = ['P', 'k']
    # fit_exponentials = np.arange(number_of_exponentials_max)+1
    # coords = {'state': states, 'fit': ,'exponential': fit_exponentials, 'parameter': fit_parameters}
    # dwell_analysis = xr.Dataset(coords=coords)
    # dwell_analysis['optimal_value'] = xr.DataArray(np.nan, dims=coords.keys(), coords=coords)
    # # dwell_analysis['error'] = xr.DataArray(np.nan, dims=('state', 'parameter'), coords={'state': positive_states, 'parameter': fit_parameters})
    # # dwell_analysis['covariance'] = xr.DataArray(np.nan, dims=('state', 'parameter','parameter'),
    # #                                    coords={'state': positive_states, 'parameter': fit_parameters, 'parameter': fit_parameters})
    #
    # # dwell_analysis.attrs['fit_function'] = fit_function.__name__
    # dwell_analysis['BIC'] = xr.DataArray(np.nan, dims=('state'), coords=coords)
    # dwell_analysis.attrs['version'] = papylio.__version__
    # dwell_analysis.attrs['method'] = method

    # dwell_analysis = {}
    # for i, state in enumerate(states):
    #     dwells_with_state = dwells.sel(dwell=dwells.state==state)
    #
    #     dwell_times = dwells_with_state.duration.values
    #     dwell_analysis_state = fit_dwell_times(dwell_times, method=method, number_of_exponentials=number_of_exponentials[state])
    #     dwell_analysis[(state, state_names[state])] = dwell_analysis_state
    #
    # dwell_analysis = pd.concat(dwell_analysis, names=('State', 'State name', 'Exponential'))
    #
    # return dwell_analysis


    dwell_analysis = []
    for i, state in enumerate(states):
        dwells_with_state = dwells.sel(dwell=dwells.state==state)

        dwell_times = dwells_with_state.duration.values
        dwell_analysis_state = fit_dwell_times(dwell_times, method=method, number_of_exponentials=number_of_exponentials[state],
                                               P_bounds=P_bounds, k_bounds=k_bounds, sampling_interval=sampling_interval,
                                               truncation=truncation, fit_dwell_times_kwargs=fit_dwell_times_kwargs)
        dwell_analysis.append(dwell_analysis_state.expand_dims({'state': [state]}))

    dwell_analysis = xr.concat(dwell_analysis, dim='state')
    if state_names is not None:
        dwell_analysis = dwell_analysis.assign_coords(state_name=('state', list(state_names.values())))

    return dwell_analysis


# [['P','k']].to_dataframe().dropna()
def plot_dwell_analysis(dwell_analysis, dwells, plot_type='pdf_binned', plot_range=None, axes=None, bins='auto_discrete',
                        log=False, sharey=True, name=None, save_path=None):
    """
    Plot the dwell time analysis results for multiple states.

    This function generates plots for the dwell time distributions of each state, showing either the probability density function (PDF) or cumulative distribution function (CDF). The results are plotted in a single figure, with options for customization, including binning, log scaling, and saving the figure.

    Parameters:
    ----------
    dwell_analysis : xarray.Dataset
        An `xarray` dataset containing the dwell time analysis results, with states as one of the dimensions.
    dwells : xarray.DataArray
        An xarray DataArray containing the dwell times and states to be plotted.
    plot_type : str or list of str, optional
        The type of plot to generate for each state. Options are 'pdf_binned' or 'cdf'. Can be a list if different types are needed for different states (default is 'pdf_binned').
    plot_range : tuple or list of tuples, optional
        The range of the plot for each state. If None, the range is set automatically (default is None).
    axes : matplotlib.Axes or array-like, optional
        The axes on which to plot the results. If None, new axes will be created (default is None).
    bins : str or int, optional
        The binning strategy for the histogram. Options are 'auto_discrete' or an integer specifying the number of bins (default is 'auto_discrete').
    log : bool, optional
        Whether to use a logarithmic scale for the y-axis (default is False).
    sharey : bool, optional
        Whether to share the y-axis across all subplots (default is True).
    name : str, optional
        The base name for the saved figure (default is None).
    save_path : pathlib.Path, optional
        The directory path where the figure will be saved (default is None, meaning the figure will not be saved).

    Returns:
    -------
    matplotlib.figure.Figure
        The figure containing the plots.
    matplotlib.Axes
        The axes containing the individual plots for each state.
    """
    # states = dwell_analysis.index.get_level_values('State').unique().values
    states = dwell_analysis.state.values

    if bins is None or isinstance(bins, numbers.Number) or isinstance(bins, str):
        bins = [bins] * len(states)

    if isinstance(plot_type, str):
        plot_type = [plot_type] * len(states)

    if plot_range is None or isinstance(plot_range[0], numbers.Number):
        plot_range = [plot_range] * len(states)

    if axes is None:
        fig, axes = plt.subplots(1,len(states), figsize=(len(states)*4.5, 4), layout='constrained', sharey=sharey)

    from collections.abc import Iterable
    if not isinstance(axes, Iterable):
        axes = [axes]

    for i, (state, dwell_analysis_state) in enumerate(dwell_analysis.groupby('state')):

        dwell_times = dwells.sel(dwell=dwells.state==state).duration.values
        plot_dwell_analysis_state(dwell_analysis_state, dwell_times, plot_type=plot_type[i], plot_range=plot_range[i], bins=bins[i], log=log, ax=axes[i])
        if 'state_name' in dwell_analysis_state.coords:
            axes[i].set_title(f'State {state}: {dwell_analysis_state.state_name.item()}')
        else:
            axes[i].set_title(f'State {state}')

        if i > 0:
            axes[i].set_ylabel('')

    if save_path is not None:
        save_path.mkdir(exist_ok=True)
        if log:
            axes[0].figure.savefig(save_path / (name + '_dwell_time_analysis_log.png'))
        else:
            axes[0].figure.savefig(save_path / (name + '_dwell_time_analysis.png'))

    return axes[0].figure, axes

#
# def plot_dwell_analysis(dwell_analysis, plot_range, axes=None, sharey=True):
#     states = dwell_analysis.index.get_level_values('State').unique().values
#     if axes is None:
#         fig, axes = plt.subplots(1,len(states), figsize=(len(states)*4, 4), layout='constrained', sharey=sharey)
#
#     for i, (state, fit_result_state) in enumerate(dwell_analysis.groupby('State')):
#         state_name = fit_result_state.index.get_level_values('State name')[0]
#         fit_result_state = fit_result_state.droplevel(level='State').droplevel(level='State name')
#         plot_dwell_analysis_state(fit_result_state, plot_range, ax=axes[i])
#         if i == 0:
#             axes[i].set_title(state_name)
#
#     return fig, axes



# if plot_range is None:
#     plot_range = (0, np.max(dwell_times))

 # plot_dwell_time_histogram(dwell_times, bins=bins, log=log, ax=ax)
