"""
Grid-forming renewable energy model with droop control and supplemantery decentralized MPC. Edited wind constraints.
"""

from andes.core import (Algeb, ConstService, ExtAlgeb, ExtService, IdxParam,
                        Lag, Model, ModelData, NumParam, State,)
from andes.core.block import GainLimiter, LagAntiWindup, PIController
from andes.core.var import AliasAlgeb, AliasState
from andes.core.service import ConstService, VarService

from andes.models.renewable.dec_helpers.mpc_func import mpc_func
from andes.models.renewable.dec_helpers.build_mpc_matrices import build_mpc_matrices
from andes.models.renewable.dec_helpers.optimal_ref import optimal_ref
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=23)   # optional, for reproducibility

class REGF1W_MPC_DECData(ModelData):
    """
    REGF1W_MPC_DEC model data.
    """

    def __init__(self):
        ModelData.__init__(self)

        self.bus = IdxParam(model='Bus',
                            info="interface bus id",
                            mandatory=True,
                            )
        self.gen = IdxParam(info="static generator index",
                            mandatory=True,
                            )
        self.Sn = NumParam(default=100.0, tex_name='S_n',
                           info='Model MVA base',
                           unit='MVA',
                           )
        self.rf = NumParam(default=0.0,
                           info="resistance",
                           z=True,
                           tex_name='r_a'
                           )
        self.xf = NumParam(default=0.2,
                           info="reactance",
                           z=True,
                           tex_name='x_s'
                           )
        self.Vdip = NumParam(default=0.8,
                             tex_name='V_{dip}',
                             info='V threshold to freeze states',
                             unit='p.u.',
                             )
        self.Tfrz = NumParam(default=0.0,
                             tex_name='T_{frz}',
                             info='Time to keep state frozen',
                             )
        self.PQFLAG = NumParam(info='P/Q priority flag; 0-Q priority, 1-P priority',
                               default=1.0,
                               unit='bool',
                               )
        self.fn = NumParam(default=60.0,
                           info="rated frequency",
                           tex_name='f',
                           )
        self.dwmax = NumParam(default=75.0,
                              info="maximum value of frequency deviation",
                              tex_name=r'\Delta \omega_{max}',
                              )
        self.dwmin = NumParam(default=-75.0,
                              info="minimum value of frequency deviation",
                              tex_name=r'\Delta \omega_{min}',
                              )
        self.wdrp = NumParam(default=0.033,
                             info="frequency droop percentage",
                             tex_name=r'\omega_{drp}',
                             )
        self.Qdrp = NumParam(default=0.045,
                             info="Voltage droop percentage",
                             tex_name='Q_{drp}',
                             )

        self.Tr = NumParam(default=0.005, tex_name='T_c',
                           info='transducer time constant',
                           unit='s',
                           )
        self.Te = NumParam(default=0.005, tex_name='T_e',
                           info='ouput state time constant',
                           unit='s',
                           )

        self.KPi = NumParam(default=0.5, tex_name='K_{Pi}',
                            info='current control proportional gain',
                            non_negative=True,
                            )
        self.KIi = NumParam(default=20.0, tex_name='K_{Ii}',
                            info='current control integral gain',
                            non_negative=True,
                            )
        self.KPv = NumParam(default=3, tex_name='K_{Pv}',
                            info='voltage control proportional gain',
                            non_negative=True,
                            )
        self.KIv = NumParam(default=10.0, tex_name='K_{Iv}',
                            info='voltage control integral gain',
                            non_negative=True,
                            )
        self.Pmax = NumParam(default=1.0, tex_name='P_{max}',
                             info='max. active power',
                             non_negative=True,
                             power=True,
                             )
        self.Pmin = NumParam(default=-1.0, tex_name='P_{min}',
                             info='min. active power',
                             power=True,
                             )

        self.KPplim = NumParam(default=5, tex_name='K_{Pplim}',
                               info='Kp for P limits',
                               non_negative=True,
                               )
        self.KIplim = NumParam(default=30.0, tex_name='K_{Iplim}',
                               info='KI for P limits',
                               non_negative=True,
                               )
        self.Qmax = NumParam(default=1.0, tex_name='Q_{max}',
                             info='max. reactive power',
                             non_negative=True,
                             power=True,
                             )
        self.Qmin = NumParam(default=-1.0, tex_name='Q_{min}',
                             info='min. reactive power',
                             power=True,
                             )
        self.KPqlim = NumParam(default=0.1, tex_name='K_{Pqlim}',
                               info='Kp for Q limits',
                               non_negative=True,
                               )
        self.KIqlim = NumParam(default=1.5, tex_name='K_{Iqlim}',
                               info='KI for Q limits',
                               non_negative=True,
                               )

        self.Tpm = NumParam(default=0.025,
                            info=r'power signal input delay (3 \Delta t)',
                            tex_name='T_{pm}'
                            )
        self.Tmpc = NumParam(default=2.0, tex_name='T_{mpc}',
                             info='MPC lag, instead of goig through PI',
                             non_negative=True,
                             )
        self.N = NumParam(default=10, tex_name='N',
                             info='MPC prediction horizon',
                             non_negative=True,
                             )
        self.mpc_step_mult = NumParam(default=4, tex_name='N',
                             info='Determines MPC time step based on simulation time step',
                             non_negative=True,
                             )
        self.Wf = NumParam(default=10, tex_name='W_f',
                             info='Weight on frequency',
                             non_negative=True,
                             )
        self.Wr = NumParam(default=10, tex_name='W_r',
                             info='Weight on RoCoF',
                             non_negative=True,
                             )
        self.Wp = NumParam(default=1, tex_name='W_p',
                             info='Weight on Power injection',
                             non_negative=True,
                             )
        self.noise_std = NumParam(default=0, tex_name='noise_std',
                             info='Measurement noise standard deviation',
                             non_negative=True,
                             )
        self.f_or_p = NumParam(default=0, tex_name='f_or_p',
                             info='Noise on frequency or power',
                             non_negative=True,
                             )
        self.loss_prob = NumParam(default=0, tex_name='loss_prob',
                             info='Probability for measurements being lost',
                             non_negative=True,
                             )
        self.gammap = NumParam(default=1.0,
                               info="P ratio of linked static gen",
                               tex_name=r'\gamma_P'
                               )
        self.gammaq = NumParam(default=1.0,
                               info="Q ratio of linked static gen",
                               tex_name=r'\gamma_Q'
                               )


class REGF1W_MPC_DECModel(Model):
    """
    Common variables and services for VSG models.
    """

    def __init__(self, system, config):
        Model.__init__(self, system, config)
        self._mpc_cache = {}
        self._ref_cache = {}
        self.psihat = {}
        self.u_store = {}
        self.f_store = {}

        self.flags.tds = True
        self.group = 'RenGen'

        self.a = ExtAlgeb(model='Bus',
                          src='a',
                          indexer=self.bus,
                          tex_name=r'\theta',
                          info='Bus voltage angle',
                          e_str='-u * Pe',
                          )
        self.v = ExtAlgeb(model='Bus',
                          src='v',
                          indexer=self.bus,
                          tex_name='V',
                          info='Bus voltage magnitude',
                          e_str='-u * Qe',
                          )

        self.p0s = ExtService(model='StaticGen',
                              src='p',
                              indexer=self.gen,
                              tex_name=r'P_{0s}',
                              info='total P of the static gen',
                              )
        self.q0s = ExtService(model='StaticGen',
                              src='q',
                              indexer=self.gen,
                              tex_name=r'Q_{0s}',
                              info='total Q of the static gen',
                              )
        self.Pref = ConstService(v_str='gammap * p0s',
                                 tex_name='P_{ref}',
                                 info='Initial P for the REGCV1 device',
                                 )
        self.Qref = ConstService(v_str='gammaq * q0s',
                                 tex_name='Q_{ref}',
                                 info='Initial Q for the REGCV1 device',
                                 )

        self.vref = ExtService(model='StaticGen',
                               src='v',
                               indexer=self.gen,
                               tex_name=r'V_{ref}',
                               info='initial v of the static gen',
                               )

        # --- Constants ---
        self.w0 = ConstService(v_str='2 * pi * fn',
                               info='rated angular frequency',
                               )

        # --- INITIALIZATION ---
        self.ixf = ConstService(v_str='1/xf',
                                tex_name=r'1/xf',
                                )
        self.Id0 = ConstService(tex_name=r'I_{d0}',
                                v_str='u * Pref / v',
                                )
        self.Iq0 = ConstService(tex_name=r'I_{q0}',
                                v_str='- u * Qref / v',
                                )

        self.vd0 = ConstService(tex_name=r'v_{d0}',
                                v_str='u * v',
                                )
        self.vq0 = ConstService(tex_name=r'v_{q0}',
                                v_str='0',
                                )
        # --- Constants end ---

        #MPC_door
        self.P_mpc = VarService(v_numeric = self.update_mpc)
        self.Pmpc = Algeb(e_str='Pmpc - P_mpc', v_str='0') # For plotting
        self.P_mpc_lag = Lag(u='P_mpc', K=1, T = self.Tmpc)

        self.Paux = Algeb(v_str='0', e_str='Paux - P_mpc_lag_y')
        self.Qaux = Algeb(v_str='0', e_str='Qaux')

        # s0 and s1
        self.Psen = Lag(u='Pe', K=1, T=self.Tr)
        self.Qsen = Lag(u='Qe', K=1, T=self.Tr)

        #Frequency measurement
        self.f = ExtAlgeb(model='BusROCOF', src='f', indexer=self.bus)
        self.rocof = ExtAlgeb(model='BusROCOF', src='Wf_y', indexer=self.bus)

        # s9 and s11
        self.Psig = Lag(u='Psen_y', K=1, T=self.Tpm)
        self.Qsig = LagAntiWindup(u='Qsen_y + Qaux', K=1, T=self.Tpm,
                                  lower=self.Qmin, upper=self.Qmax,
                                  )

        self.PIplim = PIController(u='-Psen_y + Psig_y',
                                   kp=self.KPplim, ki=self.KIplim,
                                   x0='Psen_y',
                                   )

        self.PIqlim = PIController(u='-Qsen_y + Qsig_y',
                                   kp=self.KPqlim, ki=self.KIqlim,
                                   x0='Qsen_y',
                                   )

        self.delta = State(info='virtual delta',
                           unit='rad',
                           v_str='a',
                           tex_name=r'\delta',
                           e_str='dw_y')

        # --- End reference generator loops ---

        self.vd = Algeb(tex_name='V_d',
                        info='d-axis voltage',
                        e_str='u * v * cos(delta - a) - vd',
                        v_str='vd0')

        self.vq = Algeb(tex_name='V_q',
                        info='q-axis voltage',
                        e_str='- u * v * sin(delta - a) - vq',
                        v_str='vq0')

        self.Pe = Algeb(tex_name='P_e',
                        info='active power injection from VSC',
                        e_str='vd * Id + vq * Iq - Pe',
                        v_str='Pref')
        
        self.Qe = Algeb(tex_name='Q_e',
                        info='reactive power injection from VSC',
                        e_str='- vd * Iq + vq * Id - Qe',
                        v_str='Qref')

        self.Id = Algeb(tex_name='I_d',
                        info='d-axis current',
                        v_str='Id0',
                        diag_eps=True,
                        )
        self.Iq = Algeb(tex_name='I_q',
                        info='q-axis current',
                        v_str='Iq0',
                        diag_eps=True,
                        )
        

    def update_mpc(self, **kwargs):
        u_mpc_list = np.zeros((self.n))

        # Run MPC for each inverter seperately
        for i in range(self.n):
            # Timing related variables
            k_now = float(self.system.dae.kcount)
            
            dt_sim = self.system.TDS.config.tstep
            mpc_step_mult = int(self.mpc_step_mult.v[i])
            dt_mpc = dt_sim * mpc_step_mult

            # Variables needed for building prediction and constraint matrices
            N = int(self.N.v[i])
            inv = int(self.idx.v[i])
            Wf = float(self.Wf.v[i])
            Wr = float(self.Wr.v[i])
            Wp = float(self.Wp.v[i])

            Pmax = float(self.Pmax.v[i])
            Pmin = float(self.Pmin.v[i])          
            Pref = float(self.Pref.v[i])

            noise_std = float(self.noise_std.v[i])
            f_or_p = float(self.f_or_p.v[i])
            loss_prob = float(self.loss_prob.v[i])

            #Only build the matrices when this wasn't already done
            key = (N, inv, dt_mpc, Wf, Wr, Wp, Pmax, Pref)
            if key not in self._mpc_cache:
                # build_mpc_matrices should accept N, inv, dt and return matrices
                mpc_data, ref_data = build_mpc_matrices(N, inv, dt_mpc, Wf, Wr, Wp, Pmax, Pmin, Pref)
                self._mpc_cache[key] = mpc_data
                self._ref_cache[key] = ref_data

            # Initialize cache
            if k_now == 0:
                self.psihat[(inv, k_now)] = np.zeros((5,1))
                self.u_store[inv] = 0.0
                self.mpc_switch = np.zeros((self.n,1))                      

            if k_now % mpc_step_mult == 0 and k_now != 0: # Activate MPC only at MPC time step (not simulation time step)
                # Open prediction matrices
                mpc_data = self._mpc_cache[key]
                ref_data = self._ref_cache[key]

                # Read output measurements
                freq = float((self.fn.v[i]*self.f.v[i]) - self.fn.v[i])

                if inv in self.f_store and rng.random() < loss_prob:
                    #Measurement losses
                    freq = self.f_store[inv]
                else:
                    #Normal
                    freq = freq

                self.f_store[inv] = freq

                P_c = float(self.Pe.v[i] - self.Pref.v[i])
                # Add measurement noise
                noisy_freq = freq + rng.normal(loc=0.0, scale=noise_std)
                noisy_P_c = P_c + rng.normal(loc=0.0, scale=noise_std)

                if f_or_p == 0.0:
                    y_k = np.array([[noisy_freq],[P_c]])
                elif f_or_p == 1.0:
                    y_k = np.array([[freq],[noisy_P_c]])

                # Open last state estimate
                psihat_0 = self.psihat[(inv, k_now-1)]
                
                # Update state estimate if not already updated at this time step
                kkey = (inv, k_now)
                if kkey not in self.psihat:
                    u_k = np.array([[self.u_store[inv]]])
                    psihat_next = mpc_data['A'] @ psihat_0 + mpc_data['B'] @ u_k + mpc_data['L'] @ (y_k - mpc_data['C'] @ psihat_0)
                    self.psihat[kkey] = psihat_next

                #Optimal set-point
                dhat = ref_data['vd'] @ self.psihat[kkey]
                x_ref, u_ref = optimal_ref(ref_data, dhat)
                psi_ref = np.vstack([x_ref, dhat])

                # Run the actual MPC optimization
                self.u_store[inv] = mpc_func(self.psihat[kkey], psi_ref, u_ref, N, mpc_data)
            else: # Keep MPC u and xhat constant when not at MPC time step
                if k_now != 0:
                    self.psihat[(inv, k_now)] = self.psihat[(inv, k_now-1)]
                self.u_store[inv] = self.u_store[inv]


            u_mpc_list[i] = self.u_store[inv]
        return u_mpc_list

    def v_numeric(self, **kwargs):
        """
        Disable the corresponding `StaticGen`s.
        """
        self.system.groups['StaticGen'].set(src='u', idx=self.gen.v, attr='v', value=0)


class REGF1W_MPC_DECPrimary:
    """
    Primary frequency and voltage controllers based on droop.
    """

    def __init__(self) -> None:

        self.dw = GainLimiter(u='w0 * wdrp * (PIplim_y - Psen_y + Paux)', K=1, R=1,
                              lower=self.dwmin,
                              upper=self.dwmax,
                              )

        self.vref2 = Algeb(tex_name=r'v_{ref2}',
                           info='voltage reference after droop',
                           e_str='(u * PIqlim_y - Qsen_y) * Qdrp + vref - vref2',
                           v_str='u * vref')


class REGFOuterPIModel:
    """
    Outer PI controllers for REGF1
    """

    def __init__(self, vderr: str = 'vref2 - vd', vqerr: str = '-vq'):
        self.PIvd = PIController(u=vderr,
                                 kp=self.KPv,
                                 ki=self.KIv,
                                 x0='Id0',
                                 )
        self.PIvq = PIController(u=vqerr,
                                 kp=self.KPv,
                                 ki=self.KIv,
                                 x0='Iq0',
                                 )

        self.Idref = AliasAlgeb(self.PIvd_y)
        self.Iqref = AliasAlgeb(self.PIvq_y)


class REGFInnerPIModel:
    """
    Inner current PI controllers for REGF1
    """

    def __init__(self):
        self.udref0 = ConstService(tex_name=r'u_{dref0}',
                                   v_str='vd0 + rf*Id0 - xf*Iq0'
                                   )
        self.uqref0 = ConstService(tex_name=r'u_{qref0}',
                                   v_str='vq0 + rf*Iq0 + xf*Id0',
                                   )

        # PIvd_y, PIvq_y are Idref, Iqref
        self.PIId = PIController(u='PIvd_y - Id',
                                 kp=self.KPi,
                                 ki=self.KIi,
                                 )
        self.PIIq = PIController(u='PIvq_y - Iq',
                                 kp=self.KPi,
                                 ki=self.KIi,
                                 )

        # udLag_y, uqLag_y are ud, uq
        self.Id.e_str = 'vd + Id*rf - Iq*xf - udLag_y'
        self.Iq.e_str = 'vq + Iq*rf + Id*xf - uqLag_y'

        self.udref = Algeb(tex_name=r'u_{dref}',
                           info='ud reference',
                           v_str='udref0',
                           e_str='PIId_y + vd + Id*rf - Iq*xf - udref',
                           )
        self.uqref = Algeb(tex_name=r'u_{qref}',
                           info='uq reference',
                           v_str='uqref0',
                           e_str='PIIq_y + vq + Iq*rf + Id*xf - uqref',
                           )

        self.udLag = Lag(u='udref',
                         T=self.Te,
                         K=1,
                         )
        self.uqLag = Lag(u='uqref',
                         T=self.Te,
                         K=1,
                         )

        self.ud = AliasState(self.udLag_y)
        self.uq = AliasState(self.uqLag_y)


class REGF1W_MPC_DEC(REGF1W_MPC_DECData, REGF1W_MPC_DECModel, REGF1W_MPC_DECPrimary,
            REGFOuterPIModel, REGFInnerPIModel):
    """
    Grid-forming inverter using droop.

    Implementation of EPRI Memorandum

    D. Ramasubramanian, "PROPOSAL FOR SUITE OF GENERIC GRID FORMING (GFM) POSITIVE SEQUENCE MODELS"
    """

    def __init__(self, system, config):
        REGF1W_MPC_DECData.__init__(self)

        REGF1W_MPC_DECModel.__init__(self, system, config)
        REGF1W_MPC_DECPrimary.__init__(self)
        REGFOuterPIModel.__init__(self)
        REGFInnerPIModel.__init__(self)
