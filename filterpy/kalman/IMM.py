# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-instance-attributes
"""
Created on Mon Aug  6 07:53:34 2018

@author: rlabbe
"""
from __future__ import (absolute_import, division)
import numpy as np
from numpy import dot, asarray, zeros, outer
from filterpy.common import pretty_str
from filterpy.stats import logpdf

class IMMEstimator(object):
    """ Implements an Interacting Multiple-Model (IMM) estimator.

    Parameters
    ----------

    filters : (N,) array_like of KalmanFilter objects
        List of N filters. filters[i] is the ith Kalman filter in the
        IMM estimator.

        Each filter must have the same dimension for the state `x` and `P`,
        otherwise the states of each filter cannot be mixed with each other.

    mu : (N,) array_like of float
        mode probability: mu[i] is the probability that
        filter i is the correct one.

    M : (N, N) ndarray of float
        Markov chain transition matrix. M[i,j] is the probability of
        switching from filter j to filter i.


    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.

    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.

    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    N : int
        number of filters in the filter bank

    mu : (N,) ndarray of float
        mode probability: mu[i] is the probability that
        filter i is the correct one.

    M : (N, N) ndarray of float
        Markov chain transition matrix. M[i,j] is the probability of
        switching from filter j to filter i.

    cbar : (N,) ndarray of float
        Total probability, after interaction, that the target is in state j.
        We use it as the # normalization constant.

    likelihood: (N,) ndarray of float
        Likelihood of each individual filter's last measurement.

    omega : (N, N) ndarray of float
        Mixing probabilitity - omega[i, j] is the probabilility of mixing
        the state of filter i into filter j. Perhaps more understandably,
        it weights the states of each filter by:
            x_j = sum(omega[i,j] * x_i)

        with a similar weighting for P_j


    Examples
    --------

    >>> import numpy as np
    >>> from filterpy.common import kinematic_kf
    >>> kf1 = kinematic_kf(2, 2)
    >>> kf2 = kinematic_kf(2, 2)
    >>> # do some settings of x, R, P etc. here, I'll just use the defaults
    >>> kf2.Q *= 0   # no prediction error in second filter
    >>>
    >>> filters = [kf1, kf2]
    >>> mu = [0.5, 0.5]  # each filter is equally likely at the start
    >>> trans = np.array([[0.97, 0.03], [0.03, 0.97]])
    >>> imm = IMMEstimator(filters, mu, trans)
    >>>
    >>> for i in range(100):
    >>>     # make some noisy data
    >>>     x = i + np.random.randn()*np.sqrt(kf1.R[0, 0])
    >>>     y = i + np.random.randn()*np.sqrt(kf1.R[1, 1])
    >>>     z = np.array([[x], [y]])
    >>>
    >>>     # perform predict/update cycle
    >>>     imm.predict()
    >>>     imm.update(z)
    >>>     print(imm.x.T)

    For a full explanation and more examples see my book
    Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


    References
    ----------

    Bar-Shalom, Y., Li, X-R., and Kirubarajan, T. "Estimation with
    Application to Tracking and Navigation". Wiley-Interscience, 2001.

    Crassidis, J and Junkins, J. "Optimal Estimation of
    Dynamic Systems". CRC Press, second edition. 2012.

    Labbe, R. "Kalman and Bayesian Filters in Python".
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, filters, mu, M):
        if len(filters) < 2:
            raise ValueError('filters must contain at least two filters')

        self.filters = filters
        self.mu = asarray(mu) / np.sum(mu)
        self.M = M

#        x_shape = filters[0].x.shape
#        for f in filters:
#            if x_shape != f.x.shape:
#                raise ValueError(
#                    'All filters must have the same state dimension')

        self.x = zeros(filters[0].x.shape)
        self.P = zeros(filters[0].P.shape)
        self.N = len(filters)  # number of filters
        self.likelihood = zeros(self.N)
        self.omega = zeros((self.N, self.N))
        self._compute_mixing_probabilities()

        # initialize imm state estimate based on current filters
        self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self.x_smooth = self.x.copy();
        self.P_smooth = self.P.copy()
        self.x_mixed = []
        self.P_mixed = []

    def batch_filter(self, zs):
        
        n = np.size(zs, 0)
        x_estimates = np.zeros([n,self.x.shape[0]])
        P_estimates = np.zeros([n, self.P.shape[0], self.P.shape[0]])
        model_x_posts = []
        model_x_mixeds = []
        model_x_priors = []
        model_P_posts = []
        model_P_mixeds = []
        model_P_priors = []
        likelihoods = []
        mu_store = np.zeros([n+1,len(self.filters)])
        mu_store[0,:] = self.mu
        for i, f in enumerate(self.filters):
            x_dim = f.x.shape[0]
            x_0 = np.zeros([n, x_dim])
            P_0 = np.zeros([n, x_dim, x_dim])
            
            model_x_posts.append(x_0.copy())
            model_x_mixeds.append(x_0.copy())
            model_x_priors.append(x_0.copy())
            
            model_P_posts.append(P_0.copy())
            model_P_mixeds.append(P_0.copy())
            model_P_priors.append(P_0.copy())
            
            likelihoods.append(np.zeros(n))
            
        for i in range (0,n):
            self.predict()
            for j, f in enumerate(self.filters):
                model_x_mixeds[j][i,:] = self.x_mixed[j]
                model_P_mixeds[j][i,:,:] = self.P_mixed[j]

            self.update(zs[i])
            x_estimates[i,:] = self.x;
            P_estimates[i] = self.P;
            mu_store[i+1] = self.mu;
            for j, f in enumerate(self.filters):
                model_x_priors[j][i,:] = f.x_prior
                model_P_priors[j][i,:,:] = f.P_prior
                model_x_posts[j][i,:] = f.x_post
                model_P_posts[j][i,:,:] = f.P_post
                likelihoods[j][i] = f.likelihood
                
        return(x_estimates, P_estimates,
               mu_store, 
               model_x_mixeds, model_P_mixeds, 
               model_x_priors, model_P_priors,
               model_x_posts, model_P_posts
               ,likelihoods
               )
            

        
        
        
    def update(self, z):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.
        """

        # run update on each filter, and save the likelihood
        for i, f in enumerate(self.filters):
            f.update(z)
            self.likelihood[i] = f.likelihood

        # update mode probabilities from total probability * likelihood
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        self._compute_mixing_probabilities()

        # compute mixed IMM state and covariance and save posterior estimate
        self._compute_state_estimate()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
    def batch_smooth(self, zs):
        n = np.size(zs, 0)
        (x_estimates, P_estimates,
         mu_store, 
         model_x_mixeds, model_P_mixeds, 
         model_x_priors, model_P_priors,
         model_x_posts, model_P_posts,
         model_likelihoods) = self.batch_filter(zs);
        
        x_smooth = np.zeros(x_estimates.shape);
        x_smooth[-1,:] = x_estimates[-1,:]
        P_smooth = np.zeros(P_estimates.shape);
        P_smooth[-1] = P_estimates[-1]
        
        for i in range(n-2,0,-1):
            [x_smooth[i,:], P_smooth[i]] =self._smooth_backstep(
                    modal_probs_current = mu_store[i],
                    modal_probs__smoothed_next = mu_store[i+1],
                    x_smooth_next = [x[i+1] for x in model_x_posts],
                    P_smooth_next = [x[i+1] for x in model_P_posts],
                    x_prior_next = [x[i+1] for x in model_x_priors],
                    P_prior_next = [x[i+1] for x in model_P_priors],
                    x_post = [x[i] for x in model_x_posts],
                    P_post = [x[i] for x in model_P_posts],
                    likelihoods = [x[i] for x in model_likelihoods]
                    )
        
        return x_smooth, P_smooth;
         
        
    
    def smooth(self):
        """
        
        Performs IMM smoothing one step.
        This should be performed after the update() is performed.
        
        
        """
        x_smooths, P_smooths, likelihood_smooth  = [], [], []
        mu_smooth = self.mu.copy()
        
        
        
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            Smoothing_gain = dot(f.P_post.dot(f.F.T), np.linalg.inv(f.P_prior))
            
            x_smooths.append(f.x_post + 
                             Smoothing_gain.dot(self.x_mixed[i] - f.x_prior))
            P_smooths.append(f.P_post - Smoothing_gain.dot(
                    self.P_mixed[i] - f.P_prior).dot(Smoothing_gain.T))
            
        for j, f_j in enumerate(self.filters):
            cumulation = 0.0;
            for i, f_i in enumerate(self.filters):
                #Resizing by multiplying by lots of zeros slows the program down substantially
                z = np.zeros(self.x.shape)
                x_i = self.add(x_smooths[i], z);
                x_j = self.add(f_i.x_prior, z);
                P_j = self.add(f_i.P_prior, np.zeros(self.P.shape))
                cumulation = cumulation + self.M[i,j]*logpdf(
                                              x_i, 
                                              x_j, 
                                              P_j)                                   
            likelihood_smooth.append(cumulation)
            likelihood_smooth.clear();
            
        for j, f_j in enumerate(self.filters):
            likelihood_smooth.append(f_j.likelihood)
            
        mu_smooth =  (likelihood_smooth*mu_smooth)
        mu_smooth = mu_smooth/mu_smooth.sum()
        
        self.x_smooth.fill(0)
        for i , (f, mu) in enumerate(zip(self.filters, mu_smooth)):
            self.x_smooth = self.add(self.x_smooth, x_smooths[i] * mu)

        self.P_smooth.fill(0)
        for i, (f, mu) in enumerate(zip(self.filters, mu_smooth)):
            y = self.add(f.x, - self.x_smooth)
            self.P_smooth = self.add(self.P_smooth,
                              (self.add(mu *outer(y, y) , mu *P_smooths[i])))
    
    def _smooth_backstep(self, modal_probs_current, modal_probs__smoothed_next, 
                         x_smooth_next, P_smooth_next,
                         x_prior_next, P_prior_next,
                         x_post, P_post,
                         likelihoods):
        
        mixing_probs,_ = self._compute_mixing_probabilities_functional(self.M,  modal_probs_current)
        mixing_probs,_ = self._compute_mixing_probabilities_functional(M = mixing_probs, mu = modal_probs__smoothed_next)
        
        [xs_smooth_mixed_n, Ps_smooth_mixed_n] = self._compute_mixed_estimates(x_smooth_next,
                                      P_smooth_next, 
                                      mixing_probs);
        
        xs_smooth_mixed_n = x_smooth_next;
        Ps_smooth_mixed_n = P_smooth_next;
        xs_smooth_model = []
        Ps_smooth_model = []
        for i, f in enumerate(self.filters):
            [x_temp,P_temp] = f._rts_backstep(x_post[i], P_post[i],
                 xs_smooth_mixed_n[i], Ps_smooth_mixed_n[i],
                 x_prior_next[i], P_prior_next[i])
            xs_smooth_model.append(x_temp);
            Ps_smooth_model.append(P_temp);
            
        likelihood_smooth = [];
        
        for j, f_j in enumerate(self.filters):
            cumulation = 0.0;
            for i, f_i in enumerate(self.filters):
                #Resizing by multiplying by lots of zeros slows the program down substantially
                z = np.zeros(self.x.shape)
                x_i = self.add(x_smooth_next[i], z);
                x_j = self.add(x_prior_next[i], z);
                P_j = self.add(P_prior_next[i], np.zeros(self.P.shape))
                cumulation = cumulation + self.M[i,j]*logpdf(
                                              x_i, 
                                              x_j, 
                                              P_j)                                   
            #likelihood_smooth.append(likelihoods[j])
            likelihood_smooth.append(cumulation)
        
        mu_smooth =  (likelihood_smooth*modal_probs_current)
        mu_smooth = mu_smooth/mu_smooth.sum()
        
        x_smooth_out = np.zeros(self.x.shape)
        P_smooth_out = np.zeros(self.P.shape)
        
        for i , (f, mu) in enumerate(zip(self.filters, mu_smooth)):
            x_smooth_out = self.add(x_smooth_out, xs_smooth_model[i] * mu)


        for i, (f, mu) in enumerate(zip(self.filters, mu_smooth)):
            y = self.add(f.x, - self.x_smooth)
            P_smooth_out = self.add(P_smooth_out,
                              (self.add(mu *outer(y, y) , 
                                        mu *Ps_smooth_model[i])))
        return x_smooth_out, P_smooth_out;
        
    
    def _compute_mixed_estimates(self, xs, Ps, omega):
        xs_mixed = []
        Ps_mixed = []
        
        for i, (x_input, P_input, w_row) in enumerate(zip(xs, Ps, omega)):
            x_cumilative = np.zeros(x_input.shape);
            for x_model, w in zip(xs, w_row):
                x_cumilative = self.add(x_cumilative, x_model*w)
            xs_mixed.append(x_cumilative[:x_input.shape[0]])
            
            P_cumilative = zeros(P_input.shape);
            for x_model, P_model, w in zip(xs, Ps, w_row):
                y = self.add(x_model, -x_cumilative)
                P_cumilative = self.add(P_cumilative,
                                        w*(self.add(outer(y,y), P_model)))
            Ps_mixed.append(P_cumilative[:P_input.shape[0],:P_input.shape[0]])
            
        return (xs_mixed, Ps_mixed)
    
                
            
            
        
    def predict(self, u=None):
        """
        Predict next state (prior) using the IMM state propagation
        equations.

        Parameters
        ----------

        u : np.array, optional
            Control vector. If not `None`, it is multiplied by B
            to create the control input into the system.
        """

        # compute mixed initial conditions
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            x = zeros(self.x.shape)
            for kf, wj in zip(self.filters, w):
                x = self.add(x, kf.x * wj)

            xs.append(x[:f.x.shape[0]])

            P = zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = self.add(kf.x, -x)
                P = self.add(P, wj * (
                        self.add(outer(y, y) , kf.P)))
            Ps.append(P[:f.P.shape[0],:f.P.shape[0]])

        #  compute each filter's prior using the mixed initial conditions
        self.x_mixed = xs.copy();
        self.P_mixed = Ps.copy();
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            f.x = xs[i].copy()
            f.P = Ps[i].copy()
            f.predict(u)

        # compute mixed IMM state and covariance and save posterior estimate
#        self._compute_state_estimate()
#        self.x_prior = self.x.copy()
#        self.P_prior = self.P.copy()

    def _compute_state_estimate(self):
        """
        Computes the IMM's mixed state estimate from each filter using
        the the mode probability self.mu to weight the estimates.
        """
        self.x.fill(0)
        for f, mu in zip(self.filters, self.mu):
            self.x = self.add(self.x, f.x * mu)

        self.P.fill(0)
        for f, mu in zip(self.filters, self.mu):
            y = self.add(f.x, - self.x)
            self.P = self.add(self.P,
                              (self.add(mu *outer(y, y) , mu *f.P)))
    

    def _compute_mixing_probabilities(self):
        """
        Compute the mixing probability for each filter.
        """
        
        self.cbar = dot(self.mu, self.M)
        for i in range(self.N):
            for j in range(self.N):
                self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.cbar[j]     
    
    def _compute_mixing_probabilities_functional(self,M, mu):
        """
        Compute the mixing probability for each filter.
        """
        cbar = dot(mu, M)
        omega = (M.T*mu).T/cbar
        return (omega, cbar)
                
    def __repr__(self):
        return '\n'.join([
            'IMMEstimator object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('x_post', self.x_post),
            pretty_str('P_post', self.P_post),
            pretty_str('N', self.N),
            pretty_str('mu', self.mu),
            pretty_str('M', self.M),
            pretty_str('cbar', self.cbar),
            pretty_str('likelihood', self.likelihood),
            pretty_str('omega', self.omega)
            ])
    
    def add(self, a:np.array, b:np.array):
        """
        Adds dissimalar sized arrays, adding the first available elements.
        
        e.g: add([1, 0],[0, 1, 0]) = [1, 1, 0];
        e.g: add(np.eye(2), np.eye(3)) = [[2, 0, 0],
                                            [0, 2, 0]
                                            [0, 0, 1]]
                                            
        
        This is true for 1d and 2d arrays.
        
        Parameters
        ----------

        z : np.array
            measurement for this update.
        """
        if ((a.ndim == 1) or (b.ndim == 1)):
            return self._add_available_vector(a, b);
        return self._add_available_matrix(a, b)
    
    def _add_available_vector(self, a:np.array, b:np.array):
        if len(a) < len(b):
            c = b.copy()*1.0
            c[:len(a)] += a*1.0
        else:
            c = a.copy()
            c[:len(b)] += b*1.0
        return c;
    
    def _add_available_matrix(self, a:np.array, b:np.array):
        if len(a) <= len(b):
            bigger = b;
            smaller = a;       
        else:
            bigger = a;
            smaller = b;
        
        c = bigger.copy()*1.0
        dimx, dimy = smaller.shape;
        c[:dimx,:dimy] += smaller*1.0
        return c;
