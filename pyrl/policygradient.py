from __future__ import absolute_import, division

from   collections import OrderedDict
import datetime
import os
import sys

import numpy as np

import theano
from   theano import tensor

from .         import configs, nptools, tasktools, theanotools, utils
from .debug    import DEBUG
from .networks import Networks
from .sgd      import Adam

class PolicyGradient(object):
    def __init__(self, Task, config_or_savefile, seed, dt=None, load='best'):
        self.task = Task()

        #=================================================================================
        # Network setup
        #=================================================================================

        if isinstance(config_or_savefile, str):
            #-----------------------------------------------------------------------------
            # Existing model
            #-----------------------------------------------------------------------------

            savefile = config_or_savefile
            save = utils.load(savefile)
            self.savefile = savefile
            self.save   = save
            self.config = save['config']

            # Model summary
            print("[ PolicyGradient ]")
            print("  Loading {}".format(savefile))
            print("  Last saved after {} updates.".format(save['iter']))

            # Performance
            items = OrderedDict()
            items['Best reward'] = '{} (after {} updates)'.format(save['best_reward'],
                                                                  save['best_iter'])
            if save['best_perf'] is not None:
                items.update(save['best_perf'].display(output=False))
            utils.print_dict(items)

            # Time step
            self.dt = dt
            if self.dt is None:
                self.dt = self.config['dt']
                print("Using config dt = {}".format(self.dt))

            # Leak
            alpha = self.dt/self.config['tau']

            # Which parameters to load?
            if load == 'best':
                params_p = save['best_policy_params']
                params_b = save['best_baseline_params']
            elif load == 'current':
                params_p = save['current_policy_params']
                params_b = save['current_baseline_params']
            else:
                raise ValueError(load)

            # Masks
            masks_p = save['policy_masks']
            masks_b = save['baseline_masks']

            # Policy network
            self.policy_config = save['policy_config']
            self.policy_config['alpha'] = alpha

            Network = Networks[self.config['network_type']]
            self.policy_net = Network(self.policy_config, params=params_p,
                                      masks=masks_p, name='policy')

            # Baseline network
            self.baseline_config = save['baseline_config']
            self.baseline_config['alpha'] = alpha

            Network = Networks[self.config.get('baseline_network_type',
                                               self.config['network_type'])]
            self.baseline_net = Network(self.baseline_config, params=params_b,
                                        masks=masks_b, name='baseline')
        else:
            #-----------------------------------------------------------------------------
            # Create new model.
            #-----------------------------------------------------------------------------

            config = config_or_savefile
            self.savefile = None
            self.config = config

            # Time step
            self.dt = dt
            if self.dt is None:
                self.dt = config['dt']
                print("Using config dt = {}".format(self.dt))

            # Leak
            alpha = self.dt/config['tau']

            # In continuous mode, every trial's starting state comes from the previous
            # trial's end (a plain numpy value by the time it reaches the gradient graph,
            # per run_trials), never from the `x0` shared parameter itself, so `x0` is
            # disconnected from the loss and must be excluded from `trainables` -- asking
            # Theano for its gradient otherwise raises DisconnectedInputError.
            fix          = config['fix']
            baseline_fix = config['baseline_fix']
            if config['mode'] == 'continuous':
                fix          = fix          + ['x0']
                baseline_fix = baseline_fix + ['x0']

            # Policy network
            K = config['p0']*config['N']
            self.policy_config = {
                'Nin':      config['Nin'],
                'N':        config['N'],
                'Nout':     config['Nout'],
                'p0':       config['p0'],
                'rho':      config['rho'],
                'f_hidden': config['f_hidden'],
                'gates':    config['gates'],
                'Wrec_init': config['Wrec_init'],
                'Wout':     config['Wout'],
                'f_out':    'softmax',
                'Win':      config['Win']*np.sqrt(K)/config['Nin'],
                'Win_mask': config['Win_mask'],
                'fix':      fix,
                'L2_r':     config['L2_r'],
                'L2_x':     config['L2_x'],
                'L1_Wrec':  config['L1_Wrec'],
                'L2_Wrec':  config['L2_Wrec'],
                'alpha':    alpha
                }

            # Network type
            Network = Networks[config['network_type']]
            self.policy_net = Network(self.policy_config,
                                      seed=config['policy_seed'], name='policy')

            # Baseline network
            #Win = np.zeros((self.policy_net.N + len(config['actions']), 3*config['N']))
            #Win[self.policy_net.N:] = 1

            '''
            rng = np.random.RandomState(1234)
            policy_N     = config['N']
            baseline_N   = config['N']
            baseline_Nin = self.policy_net.N + len(config['actions'])
            baseline_Win_mask = np.zeros((baseline_Nin, 3*baseline_N))
            p_in = 0.5
            baseline_Win_mask[:policy_N] = (rng.uniform(size=baseline_Win_mask[:policy_N].shape) < p_in)
            #baseline_Win = 1/np.sqrt(p_in*baseline)
            '''

            K = config['baseline_p0']*config['N']
            baseline_Nin = self.policy_net.N + len(config['actions'])
            self.baseline_config = {
                'Nin':      baseline_Nin,
                'N':        config['baseline_N'],
                'Nout':     1,
                'p0':       config['baseline_p0'],
                'rho':      config['baseline_rho'],
                'f_hidden': config['f_hidden'],
                'gates':    config['gates'],
                'Wrec_init': config['Wrec_init'],
                'Wout':     config['baseline_Wout'],
                'f_out':    'linear',
                'Win':      config['baseline_Win']*np.sqrt(K)/baseline_Nin,
                'Win_mask': config['baseline_Win_mask'],
                'bout':     config['baseline_bout'],
                'fix':      baseline_fix,
                'L2_r':     config['baseline_L2_r'],
                'L1_Wrec':  config['L1_Wrec'],
                'L2_Wrec':  config['L2_Wrec'],
                'alpha':    alpha
                }
            if self.baseline_config['bout'] is None:
                self.baseline_config['bout'] = config['R_ABORTED']

            # Network type
            Network = Networks[self.config.get('baseline_network_type',
                                               self.config['network_type'])]
            self.baseline_net = Network(self.baseline_config,
                                        seed=config['baseline_seed'], name='baseline')

        #=================================================================================
        # PG setup
        #=================================================================================

        # Network structure
        self.Nin  = self.config['Nin']
        self.N    = self.config['N']
        self.Nout = self.config['Nout']

        # Number of actions
        self.n_actions = len(self.config['actions'])

        # Recurrent noise, scaled by `2*tau/dt`
        self.scaled_var_rec = (2*self.config['tau']/self.dt) * self.config['var_rec']
        self.scaled_baseline_var_rec = ((2*self.config['tau']/self.dt)
                                        * self.config['baseline_var_rec'])

        # Run trials continuously?
        self.mode = self.config['mode']
        if self.mode == 'continuous':
            self.step_0_states = self.policy_net.func_step_0(True)

        # Maximum length of a trial
        self.Tmax = int(self.config['tmax']/self.config['dt']) + 1

        # Discount future reward
        if np.isfinite(self.config['tau_reward']):
            self.alpha_reward = self.dt/self.config['tau_reward']
            def discount_factor(t):
                return np.exp(-t*self.alpha_reward)
        else:
            def discount_factor(t):
                return 1
        self.discount_factor = discount_factor

        # Reward on aborted trials
        self.abort_on_last_t = self.config.get('abort_on_last_t', True)
        if 'R_TERMINAL' in self.config and self.config['R_TERMINAL'] is not None:
            self.R_TERMINAL = self.config['R_TERMINAL']
        else:
            self.R_TERMINAL = self.config['R_ABORTED']
        self.R_ABORTED = self.config['R_ABORTED']

        # Random number generators, one per independent stream. `self.rng` keeps its name
        # and is the stimulus/trial stream, which is what every analysis module reads.
        # Savefiles written before the streams were split carry neither derived seed, so
        # fall back to deriving them from `seed`.
        derived = configs.derive_seeds(seed, self.config)
        self.rng        = nptools.get_rng(seed, __name__ + ' (trials)')
        self.rng_action = nptools.get_rng(self.config.get('action_rng_seed',
                                                          derived['action_rng_seed']),
                                          __name__ + ' (actions)')
        self.rng_noise  = nptools.get_rng(self.config.get('noise_rng_seed',
                                                          derived['noise_rng_seed']),
                                          __name__ + ' (noise)')

        # Compile functions
        self.policy_step_0   = self.policy_net.func_step_0()
        self.policy_step_t   = self.policy_net.func_step_t()
        self.baseline_step_0 = self.baseline_net.func_step_0()
        self.baseline_step_t = self.baseline_net.func_step_t()

        # Performance
        self.Performance = self.config['Performance']

    def checkpoint_dir(self):
        if self.savefile is None:
            raise ValueError("no savefile, so there is no checkpoint directory")
        return os.path.join(os.path.dirname(self.savefile), 'checkpoints')

    def list_checkpoints(self):
        d = self.checkpoint_dir()
        return sorted(int(f[5:-4]) for f in os.listdir(d)
                      if f.startswith('iter_') and f.endswith('.pkl'))

    def load_checkpoint(self, iter_):
        """
        Swap in a checkpoint's parameters in place. The network structure comes from the
        savefile; only the parameters are overridden, and since they are shared variables
        the already-compiled Theano functions need not be rebuilt.

        """
        ckpt = utils.load(os.path.join(self.checkpoint_dir(),
                                       'iter_{:06d}.pkl'.format(int(iter_))))
        self.policy_net.set_values(ckpt['policy_params'])
        self.baseline_net.set_values(ckpt['baseline_params'])

        return ckpt

    def make_noise(self, size, var=0):
        if var > 0:
            return theanotools.asarray(self.rng_noise.normal(scale=np.sqrt(var),
                                                             size=size))
        return theanotools.zeros(size)

    def run_trials(self, trials, init=None, init_b=None,
                   return_states=False, perf=None, task=None, progress_bar=False,
                   p_dropout=0):
        if isinstance(trials, list):
            n_trials = len(trials)
        else:
            n_trials = trials
            trials   = []

        if return_states:
            run_value_network = True
        else:
            run_value_network = False

        # Storage
        U   = theanotools.zeros((self.Tmax, n_trials, self.Nin))
        Z   = theanotools.zeros((self.Tmax, n_trials, self.Nout))
        A   = theanotools.zeros((self.Tmax, n_trials, self.n_actions))
        R   = theanotools.zeros((self.Tmax, n_trials))
        M   = theanotools.zeros((self.Tmax, n_trials))
        Z_b = theanotools.zeros((self.Tmax, n_trials))

        # Noise
        Q   = self.make_noise((self.Tmax, n_trials, self.policy_net.noise_dim),
                               self.scaled_var_rec)
        Q_b = self.make_noise((self.Tmax, n_trials, self.baseline_net.noise_dim),
                               self.scaled_baseline_var_rec)

        x_t   = theanotools.zeros((1, self.policy_net.N))
        x_t_b = theanotools.zeros((1, self.baseline_net.N))

        # Dropout mask
        #D   = np.ones((self.Tmax, n_trials, self.policy_net.N))
        #D_b = np.ones((self.Tmax, n_trials, self.baseline_net.N))
        #if p_dropout > 0:
        #    D   -= (np.uniform(size=D.shape) < p_dropout)
        #    D_b -= (np.uniform(size=D_b.shape) < p_dropout)

        # Firing rates
        if return_states:
            r_policy = theanotools.zeros((self.Tmax, n_trials, self.policy_net.N))
            r_value  = theanotools.zeros((self.Tmax, n_trials, self.baseline_net.N))

        # Keep track of initial conditions
        if self.mode == 'continuous':
            x0   = theanotools.zeros((n_trials, self.policy_net.N))
            x0_b = theanotools.zeros((n_trials, self.baseline_net.N))
        else:
            x0   = None
            x0_b = None

        # Performance
        if perf is None:
            perf = self.Performance()

        # Setup progress bar
        if progress_bar:
            progress_inc  = max(int(n_trials/50), 1)
            progress_half = 25*progress_inc
            if progress_half > n_trials:
                progress_half = -1
            utils.println("[ PolicyGradient.run_trials ] ")

        # Outcome of the previous trial in this batch, as (choice, terminal reward), for
        # tasks that feed trial history back as input. None for the first trial of every
        # batch and after any trial that ended without a decision; tasks that don't read
        # `trial['prev_outcome']` are unaffected.
        prev_outcome = None

        for n in xrange(n_trials):
            if progress_bar and n % progress_inc == 0:
                if n == 0:
                    utils.println("0")
                elif n == progress_half:
                    utils.println("50")
                else:
                    utils.println("|")

            # Initialize trial
            if hasattr(self.task, 'start_trial'):
                self.task.start_trial()

            # Generate trials
            if n < len(trials):
                trial = trials[n]
            else:
                trial = self.task.get_condition(self.rng, self.dt)
                trials.append(trial)
            trial['prev_outcome'] = prev_outcome

            #-----------------------------------------------------------------------------
            # Time t = 0
            #-----------------------------------------------------------------------------

            t = 0
            if init is None:
                z_t,   x_t[0]   = self.policy_step_0()
                z_t_b, x_t_b[0] = self.baseline_step_0()
            else:
                z_t,   x_t[0]   = init
                z_t_b, x_t_b[0] = init_b
            Z[t,n]   = z_t
            Z_b[t,n] = z_t_b

            # Save initial condition
            if x0 is not None:
                x0[n]   = x_t[0]
                x0_b[n] = x_t_b[0]

            # Save states
            if return_states:
                r_policy[t,n] = self.policy_net.firing_rate(x_t[0])
                r_value[t,n]  = self.baseline_net.firing_rate(x_t_b[0])

            # Select action
            a_t = theanotools.choice(self.rng_action, self.Nout,
                                     p=np.reshape(z_t, (self.Nout,)))
            A[t,n,a_t] = 1

            #a_t = self.rng.normal(np.reshape(z_t, (self.Nout,)), self.sigma)
            #A[t,n,0] = a_t

            # Trial step. `reward_t` keeps the undiscounted reward, since R[t,n] is scaled
            # by the discount factor in place below.
            U[t,n], reward_t, status = self.task.get_step(self.rng, self.dt,
                                                          trial, t+1, a_t)
            R[t,n] = reward_t
            u_t    = U[t,n]
            M[t,n] = 1

            # Noise
            q_t   = Q[t,n]
            q_t_b = Q_b[t,n]

            #-----------------------------------------------------------------------------
            # Time t > 0
            #-----------------------------------------------------------------------------

            for t in xrange(1, self.Tmax):
                # Aborted episode
                if not status['continue']:
                    break

                # Policy
                z_t, x_t[0] = self.policy_step_t(u_t[None,:], q_t[None,:], x_t)
                Z[t,n] = z_t

                # Baseline
                r_t = self.policy_net.firing_rate(x_t[0])
                u_t_b = np.concatenate((r_t, A[t-1,n]), axis=-1)
                z_t_b, x_t_b[0] = self.baseline_step_t(u_t_b[None,:],
                                                       q_t_b[None,:],
                                                       x_t_b)
                Z_b[t,n] = z_t_b

                # Firing rates
                if return_states:
                    r_policy[t,n] = self.policy_net.firing_rate(x_t[0])
                    r_value[t,n]  = self.baseline_net.firing_rate(x_t_b[0])

                    #W = self.policy_net.get_values()['Wout']
                    #b = self.policy_net.get_values()['bout']
                    #V = r_policy[t,n].dot(W) + b
                    #print(t)
                    #print(V)
                    #print(np.exp(V))

                # Select action
                a_t = theanotools.choice(self.rng_action, self.Nout,
                                         p=np.reshape(z_t, (self.Nout,)))
                A[t,n,a_t] = 1

                #a_t = self.rng.normal(np.reshape(z_t, (self.Nout,)), self.sigma)
                #A[t,n,0] = a_t

                # Trial step
                if self.abort_on_last_t and t == self.Tmax-1:
                    U[t,n]   = 0
                    reward_t = self.R_TERMINAL
                    R[t,n]   = reward_t
                    status   = {'continue': False, 'reward': reward_t}
                else:
                    U[t,n], reward_t, status = self.task.get_step(self.rng, self.dt,
                                                                  trial, t+1, a_t)
                    R[t,n] = reward_t
                R[t,n] *= self.discount_factor(t)

                u_t    = U[t,n]
                M[t,n] = 1

                # Noise
                q_t   = Q[t,n]
                q_t_b = Q_b[t,n]

            #-----------------------------------------------------------------------------

            # Update performance. `action`/`reward` are the trial's final action index and
            # its undiscounted terminating reward, recorded for every trial.
            status['action'] = a_t
            status['reward'] = reward_t
            perf.update(trial, status)

            # Outcome handed to the next trial in this batch. A trial that ended without a
            # decision (no choice) passes None, so its successor sees no history.
            if status.get('choice') is not None:
                prev_outcome = (status['choice'], reward_t)
            else:
                prev_outcome = None
            # Hand the outcome to the task, for closed-loop schedules such as the lab's
            # anti-bias (examples/models/pagan_hist_antibias.py). Added 11 Oct 2026; tasks
            # without the hook are unaffected.
            if hasattr(self.task, 'record_outcome'):
                self.task.record_outcome(trial, status, reward_t)

            # Save next state if necessary
            if self.mode == 'continuous':
                init   = self.policy_step_t(u_t[None,:], q_t[None,:], x_t)
                init_b = self.baseline_step_t(u_t_b[None,:], q_t_b[None,:], x_t_b)
        if progress_bar:
            print("100")

        #---------------------------------------------------------------------------------

        rvals = [U, Q, Q_b, Z, Z_b, A, R, M, init, init_b, x0, x0_b, perf]
        if return_states:
            rvals += [r_policy, r_value]

        return rvals

    def func_update_policy(self, Tmax, use_x0=False, accumulators=None):
        U = tensor.tensor3('U') # Inputs
        Q = tensor.tensor3('Q') # Noise

        if use_x0:
            x0_ = tensor.matrix('x0_')
        else:
            x0  = self.policy_net.params['x0']
            x0_ = tensor.alloc(x0, U.shape[1], x0.shape[0])

        log_z_0  = self.policy_net.get_outputs_0(x0_, log=True)
        r, log_z = self.policy_net.get_outputs(U, Q, x0_, log=True)

        # Learning rate
        lr = tensor.scalar('lr')

        A = tensor.tensor3('A')
        R = tensor.matrix('R')
        b = tensor.matrix('b')
        M = tensor.matrix('M')

        # Mdec masks the REINFORCE score function to the decision epoch: for tasks (like
        # pagan) where actions before some `trial['dec_t0']` are null and cannot affect the
        # trial, giving them credit for the eventual return would just add variance to the
        # gradient. Mdec == M for tasks with no such epoch (e.g. mante), so this is a no-op
        # there. Only logpi_0/logpi_t are masked with it - entropy and the regularizers
        # below still use the full-trial M.
        Mdec = tensor.matrix('Mdec')

        logpi_0 = tensor.sum(log_z_0*A[0], axis=-1)*Mdec[0]
        logpi_t = tensor.sum(log_z*A[1:],  axis=-1)*Mdec[1:]

        # Entropy. entropy_0/entropy_t hold sum(p*log p) (negative entropy), so adding
        # beta_entropy*entropy to the (minimized) objective below is equivalent to
        # subtracting beta_entropy*true_entropy, i.e. rewarding higher entropy.
        entropy_0 = tensor.sum(tensor.exp(log_z_0)*log_z_0, axis=-1)*M[0]
        entropy_t = tensor.sum(tensor.exp(log_z)*log_z, axis=-1)*M[1:]
        entropy   = (tensor.sum(entropy_0) + tensor.sum(entropy_t))/tensor.sum(M)

        #def f(x):
        #    return -x**2/2/self.sigma**2

        #logpi_0 = tensor.sum(f(A[0] - z_0), axis=-1)*M[0]
        #logpi_t = tensor.sum(f(A[1:] - z), axis=-1)*M[1:]

        # Enforce causality
        Mcausal = theanotools.zeros((Tmax-1, Tmax-1))
        for i in xrange(Mcausal.shape[0]):
            Mcausal[i,i:] = 1
        Mcausal = theanotools.shared(Mcausal, 'Mcausal')

        J0 = logpi_0*R[0]
        J0 = tensor.mean(J0)
        J  = (logpi_t.T).dot(Mcausal).dot(R[1:]*M[1:])
        J  = tensor.nlinalg.trace(J)/J.shape[0]

        J += J0

        # Second term
        Jb0 = logpi_0*b[0]
        Jb0 = tensor.mean(Jb0)
        Jb  = logpi_t*b[1:]
        Jb  = tensor.mean(tensor.sum(Jb, axis=0))

        J -= Jb0 + Jb

        # Objective function
        obj = -J + self.policy_net.get_regs(x0_, r, M) + self.config['beta_entropy']*entropy

        # SGD
        self.policy_sgd = Adam(self.policy_net.trainables, accumulators=accumulators)
        if self.policy_net.type == 'simple':
            i = self.policy_net.index('Wrec')
            grads = tensor.grad(obj, self.policy_net.trainables)
            grads[i] += self.policy_net.get_dOmega_dWrec(-J, r)
            norm, grads, updates = self.policy_sgd.get_updates(obj, lr, grads=grads)
        else:
            norm, grads, updates = self.policy_sgd.get_updates(obj, lr)

        if use_x0:
            args = [x0_]
        else:
            args = []
        args += [U, Q, A, R, b, M, Mdec, lr]

        return theano.function(args, norm, updates=updates)

    def func_update_baseline(self, use_x0=False, accumulators=None):
        U  = tensor.tensor3('U')
        R  = tensor.matrix('R')
        R_ = R.reshape((R.shape[0], R.shape[1], 1))
        Q  = tensor.tensor3('Q')

        if use_x0:
            x0_ = tensor.matrix('x0_')
        else:
            x0  = self.baseline_net.params['x0']
            x0_ = tensor.alloc(x0, U.shape[1], x0.shape[0])

        z_0   = self.baseline_net.get_outputs_0(x0_)
        r, z  = self.baseline_net.get_outputs(U, Q, x0_)
        z_all = tensor.concatenate([z_0.reshape((1, z_0.shape[0], z_0.shape[1])), z],
                                   axis=0)

        # Learning rate
        lr = tensor.scalar('lr')

        # Reward prediction error
        M    = tensor.matrix('M')
        L2   = tensor.sum((tensor.sqr(z_all[:,:,0] - R))*M)/tensor.sum(M)
        RMSE = tensor.sqrt(L2)

        # Objective function
        obj = L2 + self.baseline_net.get_regs(x0_, r, M)

        # SGD
        self.baseline_sgd = Adam(self.baseline_net.trainables, accumulators=accumulators)
        if self.baseline_net.type == 'simple':
            i = self.baseline_net.index('Wrec')
            grads = tensor.grad(obj, self.baseline_net.trainables)
            grads[i] += self.baseline_net.get_dOmega_dWrec(L2, r)
            norm, grads, updates = self.baseline_sgd.get_updates(obj, lr, grads=grads)
        else:
            norm, grads, updates = self.baseline_sgd.get_updates(obj, lr)

        if use_x0:
            args = [x0_]
        else:
            args = []
        args += [U, Q, R, M, lr]

        return theano.function(args, [z_all[:,:,0], norm, RMSE], updates=updates)

    def train(self, savefile, recover=False):
        """
        Train network.

        """
        #=================================================================================
        # Parameters
        #=================================================================================

        max_iter     = self.config['max_iter']
        lr           = self.config['lr']
        baseline_lr  = self.config['baseline_lr']
        n_gradient   = self.config['n_gradient']
        n_validation = self.config['n_validation']
        checkfreq    = self.config['checkfreq']

        if self.mode == 'continuous':
            print("[ PolicyGradient.train ] Continuous mode.")
            use_x0 = True
        else:
            use_x0 = False

        # GPU?
        if theanotools.get_processor_type() == 'gpu':
            gpu = 'yes'
        else:
            gpu = 'no'

        # Print settings
        items = OrderedDict()
        items['GPU']                      = gpu
        items['Network type (policy)']    = self.config['network_type']
        items['Network type (baseline)']  = self.config.get('baseline_network_type',
                                                            self.config['network_type'])
        items['N (policy)']               = self.config['N']
        items['N (baseline)']             = self.config['baseline_N']
        items['Conn. prob. (policy)']     = self.config['p0']
        items['Conn. prob. (baseline)']   = self.config['baseline_p0']
        items['f_hidden']                 = self.config['f_hidden']
        items['Gates']                    = self.config['gates']
        items['Wrec_init']                = self.config['Wrec_init']
        items['rho (policy)']             = self.config['rho']
        items['rho (baseline)']           = self.config['baseline_rho']
        items['dt']                       = str(self.dt) + ' ms'
        items['tau']                      = str(self.config['tau']) + ' ms'
        items['alpha']                    = self.dt/self.config['tau']
        items['tau_reward']               = str(self.config['tau_reward']) + ' ms'
        items['var_rec (policy)']         = self.config['var_rec']
        items['var_rec (baseline)']       = self.config['baseline_var_rec']
        items['scaled var_rec (policy)']  = self.scaled_var_rec
        items['L2_x (policy)']            = self.config['L2_x']
        items['beta_entropy']             = self.config['beta_entropy']
        items['Learning rate (policy)']   = self.config['lr']
        items['Learning rate (baseline)'] = self.config['baseline_lr']
        items['Seed (weight init)']       = '{} -> policy {}, baseline {}'.format(
            self.config.get('weight_seed'), self.config.get('policy_seed'),
            self.config.get('baseline_seed'))
        items['Seed (stimulus)']          = '{} -> rng {}'.format(
            self.config.get('stim_seed'), self.config.get('seed'))
        items['Seed (action)']            = '{} -> rng {}, noise {}'.format(
            self.config.get('action_seed'), self.config.get('action_rng_seed'),
            self.config.get('noise_rng_seed'))
        items['Max time steps']           = self.Tmax
        items['Num. trials (gradient)']   = self.config['n_gradient']
        items['Num. trials (validation)'] = self.config['n_validation']
        utils.print_dict(items)

        #=================================================================================
        # Setup
        #=================================================================================

        if recover:
            print("Resume training.")
            update_policy   = self.func_update_policy(self.Tmax, use_x0,
                                                      accumulators=self.save['net_sgd'])
            update_baseline = self.func_update_baseline(use_x0,
                                                        accumulators=self.save['baseline_sgd'])

            # Resume training from here
            iter_start = self.save['iter']
            print("Last saved was after {} updates.".format(self.save['iter']))

            # Random number generator
            print("Resetting RNG state.")
            self.rng.set_state(self.save['rng_state'])
            if 'rng_action_state' in self.save:
                self.rng_action.set_state(self.save['rng_action_state'])
                self.rng_noise.set_state(self.save['rng_noise_state'])

            # Keep track of best results
            best_iter            = self.save['best_iter']
            best_reward          = self.save['best_reward']
            best_perf            = self.save['best_perf']
            best_params          = self.save['best_policy_params']
            best_baseline_params = self.save['best_baseline_params']

            # Initial states
            init   = self.save['init']
            init_b = self.save['init_b']

            # Training history
            perf             = self.save['perf']
            training_history = self.save['training_history']
            trials_tot       = self.save['trials_tot']
        else:
            update_policy   = self.func_update_policy(self.Tmax, use_x0)
            update_baseline = self.func_update_baseline(use_x0)

            # Start training from here
            iter_start = 0

            # Keep track of best results
            best_iter   = -1
            best_reward = -np.inf
            best_perf   = None
            best_params = self.policy_net.get_values()
            best_baseline_params = self.baseline_net.get_values()

            # Initial states
            init   = None
            init_b = None

            # Performance history
            perf             = None
            training_history = []
            trials_tot       = 0

        #=================================================================================
        # Train
        #=================================================================================

        if hasattr(self.task, 'start_session'):
            self.task.start_session(self.rng)

        grad_norms_policy   = []
        grad_norms_baseline = []

        tstart = datetime.datetime.now()
        try:
            for iter_ in xrange(iter_start, max_iter+1):
                if iter_ % checkfreq == 0 or iter_ == max_iter:
                    if hasattr(self.task, 'n_validation'):
                        n_validation = self.task.n_validation
                    if n_validation > 0:
                        #-----------------------------------------------------------------
                        # Validation
                        #-----------------------------------------------------------------

                        # Report
                        elapsed = utils.elapsed_time(tstart)
                        print("After {} updates ({})".format(iter_, elapsed))

                        # RNG state
                        rng_state        = self.rng.get_state()
                        rng_action_state = self.rng_action.get_state()
                        rng_noise_state  = self.rng_noise.get_state()

                        # Trials. Tasks with a closed-loop schedule (pagan_hist_antibias)
                        # are told this is validation, so the batch is drawn 50/50 and not
                        # counted (hook added 11 Oct 2026).
                        if hasattr(self.task, 'set_validation'):
                            self.task.set_validation(True)
                        trials = [self.task.get_condition(self.rng, self.dt)
                                  for i in xrange(n_validation)]

                        # Run trials
                        (U, Q, Q_b, Z, Z_b, A, R, M, init_, init_b_, x0_, x0_b_,
                         perf_) = self.run_trials(trials, progress_bar=True)
                        if hasattr(self.task, 'set_validation'):
                            self.task.set_validation(False)
                        if hasattr(self.task, 'update'):
                            self.task.update(perf_)

                        # Termination condition
                        terminate = False
                        if hasattr(self.task, 'terminate'):
                            if self.task.terminate(perf_):
                                terminate = True

                        # Save
                        mean_reward = np.sum(R*M)/n_validation
                        record = {
                            'iter':        iter_,
                            'mean_reward': mean_reward,
                            'n_trials':    trials_tot,
                            'perf':        perf_
                            }
                        if mean_reward > best_reward or terminate:
                            best_iter   = iter_
                            best_reward = mean_reward
                            best_perf   = perf_
                            best_params          = self.policy_net.get_values()
                            best_baseline_params = self.baseline_net.get_values()

                            record['new_best'] = True
                            training_history.append(record)
                        else:
                            record['new_best'] = False
                            training_history.append(record)

                        # Save
                        save = {
                            'iter':                    iter_,
                            'config':                  self.config,
                            'policy_config':           self.policy_net.config,
                            'baseline_config':         self.baseline_net.config,
                            'policy_masks':            self.policy_net.get_masks(),
                            'baseline_masks':          self.baseline_net.get_masks(),
                            'current_policy_params':   self.policy_net.get_values(),
                            'current_baseline_params': self.baseline_net.get_values(),
                            'best_iter':               best_iter,
                            'best_reward':             best_reward,
                            'best_perf':               best_perf,
                            'best_policy_params':      best_params,
                            'best_baseline_params':    best_baseline_params,
                            'rng_state':               rng_state,
                            'rng_action_state':        rng_action_state,
                            'rng_noise_state':         rng_noise_state,
                            'init':                    init,
                            'init_b':                  init_b,
                            'perf':                    perf,
                            'training_history':        training_history,
                            'trials_tot':              trials_tot,
                            'net_sgd':                 self.policy_sgd.get_values(),
                            'baseline_sgd':            self.baseline_sgd.get_values()
                            }
                        utils.save(savefile, save)

                        # Full-parameter checkpoint snapshot (not overwritten each time),
                        # so the entire training trajectory is preserved for later analysis
                        # or hyperparameter tuning, not just the current/best params above.
                        checkpoint_dir = os.path.join(os.path.dirname(savefile), 'checkpoints')
                        utils.mkdir_p(checkpoint_dir)
                        checkpoint_file = os.path.join(checkpoint_dir,
                                                       'iter_{:06d}.pkl'.format(iter_))
                        utils.save(checkpoint_file, {
                            'iter':            iter_,
                            'trials_tot':      trials_tot,
                            'config':          self.config,
                            'policy_params':   self.policy_net.get_values(),
                            'baseline_params': self.baseline_net.get_values(),
                            'mean_reward':     mean_reward,
                            'perf':            perf_
                            })

                        # Reward
                        items = OrderedDict()
                        items['Best reward'] = '{} (iteration {})'.format(best_reward,
                                                                          best_iter)
                        items['Mean reward'] = '{}'.format(mean_reward)

                        # Performance
                        if perf_ is not None:
                            items.update(perf_.display(output=False))

                        # Value prediction error
                        V = np.zeros_like(R)
                        for k in xrange(V.shape[0]):
                            V[k] = np.sum(R[k:]*M[k:], axis=0)
                        error = np.sqrt(np.sum((Z_b - V)**2*M)/np.sum(M))
                        items['Prediction error'] = '{}'.format(error)

                        # Mean softmax action probability over valid (t, trial) steps, one
                        # number per action, in the order given by config['actions']. Z is
                        # exp(log-softmax), i.e. already a probability, not a raw logit.
                        M_action = (np.tile(M.T, (Z.shape[-1], 1, 1))).T
                        mean_action_probs = (Z*M_action).sum(axis=(0, 1))/np.sum(M)
                        action_names = sorted(self.config['actions'],
                                              key=self.config['actions'].get)
                        items['P(action)'] = ', '.join(
                            '{}={:.4f}'.format(name, p)
                            for name, p in zip(action_names, mean_action_probs))

                        # Gradient norms
                        if len(grad_norms_policy) > 0:
                            if DEBUG:
                                items['|grad| (policy)']   = [len(grad_norms_policy)] + [f(grad_norms_policy)
                                                              for f in [np.min, np.median, np.max]]
                                items['|grad| (baseline)'] = [len(grad_norms_baseline)] + [f(grad_norms_baseline)
                                                              for f in [np.min, np.median, np.max]]
                            grad_norms_policy   = []
                            grad_norms_baseline = []

                        # Print
                        utils.print_dict(items)

                        # Target reward reached
                        if best_reward >= self.config['target_reward']:
                            print("Target reward reached.")
                            return

                        # Terminate
                        if terminate:
                            print("Termination criterion satisfied.")
                            return
                    else:
                        '''
                        #-----------------------------------------------------------------
                        # Ongoing learning
                        #-----------------------------------------------------------------

                        if not training_history:
                            training_history.append(perf)
                        if training_history[0] is None:
                            training_history[0] = perf

                        # Save
                        save = {
                            'iter':                    iter,
                            'config':                  self.config,
                            'policy_config':           self.policy_net.config,
                            'baseline_config':         self.baseline_net.config,
                            'masks_p':                 self.policy_net.get_masks(),
                            'masks_b':                 self.baseline_net.get_masks(),
                            'current_policy_params':   self.policy_net.get_values(),
                            'current_baseline_params': self.baseline_net.get_values(),
                            'rng_state':               self.rng.get_state(),
                            'rng_action_state':        self.rng_action.get_state(),
                            'rng_noise_state':         self.rng_noise.get_state(),
                            'init':                    init,
                            'init_b':                  init_b,
                            'perf':                    perf,
                            'training_history':        training_history,
                            'trials_tot':              trials_tot,
                            'net_sgd':                 self.policy_sgd.get_values(),
                            'baseline_sgd':            self.baseline_sgd.get_values()
                            }
                        utils.save(savefile, save)
                        '''
                        if perf is not None:
                            perf.display()

                        # Termination condition
                        terminate = False
                        if hasattr(self.task, 'terminate'):
                            if perf is not None and self.task.terminate(perf):
                                terminate = True
                        '''
                        # Report
                        if iter % 100 == 1:
                            elapsed = utils.elapsed_time(tstart)
                            print("After {} updates ({})".format(iter, elapsed))
                            if perf is not None:
                                perf.display()
                        '''
                        # Terminate
                        if terminate:
                            print("Termination criterion satisfied.")
                            return

                if iter_ == max_iter:
                    print("Reached maximum number of iterations ({}).".format(iter_))
                    sys.exit(0)

                #-------------------------------------------------------------------------
                # Run trials
                #-------------------------------------------------------------------------

                # Trial conditions
                if hasattr(self.task, 'n_gradient'):
                    n_gradient = self.task.n_gradient
                trials = [self.task.get_condition(self.rng, self.dt)
                          for i in xrange(n_gradient)]

                # A session is exactly one training batch: in continuous mode, carried
                # state resets to x0 at the start of every session (not just once, at the
                # start of training), so that a session is always the same fixed-length
                # chain of n_gradient trials and carried state never spans a weight update.
                if self.mode == 'continuous':
                    init, init_b = None, None

                # Run trials
                (U, Q, Q_b, Z, Z_b, A, R, M, init, init_b, x0, x0_b,
                 perf, r_policy, r_value) = self.run_trials(trials,
                                                            init=init, init_b=init_b,
                                                            return_states=True, perf=perf)

                #-------------------------------------------------------------------------
                # Update baseline
                #-------------------------------------------------------------------------

                baseline_inputs = np.concatenate((r_policy, A), axis=-1)

                # Compute return
                R_b = np.zeros_like(R)
                for k in xrange(R.shape[0]):
                    R_b[k] = np.sum(R[k:]*M[k:], axis=0)

                if use_x0:
                    args = [x0_b]
                else:
                    args = []
                args += [baseline_inputs[:-1], Q_b, R_b, M, baseline_lr]
                b, norm_b, rmse = update_baseline(*args)
                #print("Prediction error = {}".format(rmse))

                norm_b = float(norm_b)
                #print("norm_b = {}".format(norm_b))
                if np.isfinite(norm_b):
                    grad_norms_baseline.append(float(norm_b))

                #-------------------------------------------------------------------------
                # Update policy
                #-------------------------------------------------------------------------

                # Mdec == M except that steps before trial['dec_t0'] are zeroed, so the
                # REINFORCE score function only sees decision-epoch actions. Tasks with no
                # 'dec_t0' (dec_t0 defaults to 0) are unaffected.
                Mdec = np.zeros_like(M)
                for k, trial in enumerate(trials):
                    dec_t0 = trial.get('dec_t0', 0)
                    Mdec[dec_t0:, k] = M[dec_t0:, k]

                if use_x0:
                    args = [x0]
                else:
                    args = []
                args += [U[:-1], Q, A, R, b, M, Mdec, lr]
                norm = update_policy(*args)

                norm = float(norm)
                #print("norm = {}".format(norm))
                if np.isfinite(norm):
                    grad_norms_policy.append(norm)

                trials_tot += n_gradient

        except KeyboardInterrupt:
            print("Training interrupted by user during iteration {}.".format(iter_))
            sys.exit(0)
