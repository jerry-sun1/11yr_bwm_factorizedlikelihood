#this script now

from __future__ import division

import numpy as np
import os, glob, json
import matplotlib.pyplot as plt
import pickle
import scipy.linalg as sl
import healpy as hp
import multiprocessing as mp

from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
from enterprise import constants as const
from enterprise.signals.signal_base import LogLikelihood
from enterprise.signals.signal_base import LookupLikelihood

from enterprise_extensions import models as ee_models
from enterprise_extensions import model_utils as ee_model_utils
from enterprise_extensions import sampler as ee_sampler


from la_forge.core import Core, load_Core
from la_forge import rednoise
from la_forge.diagnostics import plot_chains

'''
This function gets the polarization mtx from the sky location and polarization of the burst.
The details are worked out in a jupyter notebook that Jerry has
We're going to start by assuming that the burst is just one polarization.
@params:
    theta: polar angle of the sky-location of the burst
    phi: azimuthal angle of the sky-location of the burst
    psi: polarization angle of the burst

@return:
    rotated_polarization: the polarization tensor of a plus-polarized wave at some sky-location theta,phi
'''
def get_polarization_mtx(theta, phi, psi):
    unrotated_polarization = np.matrix([[1,0,0],[0,-1,0],[0,0,0]])

    Ry_skyloc = [[np.cos(theta), 0, np.sin(theta)],[0,1,0],[-1*np.sin(theta), 0, np.cos(theta)]]
    Rz_pol = [[np.cos(psi) , -1*np.sin(psi) , 0],[np.sin(psi), np.cos(psi), 0],[0,0,1]]
    Rz_skyloc =[[np.cos(phi) , -1*np.sin(phi) , 0],[np.sin(phi), np.cos(phi), 0],[0,0,1]]

    right_mess = np.matmul(np.matmul(Rz_skyloc, Ry_skyloc), Rz_pol)
    left_mess = np.transpose(right_mess)

    left_messed = np.matmul(left_mess, unrotated_polarization)
    rotated_polarization = np.matmul(left_messed, right_mess)

    return rotated_polarization


'''
Utility function to interpolate the probability using a histogram using nearest neighbors
@params:
    hist: The histogram to look at
    xedges: The bin edges of the histogram
    val: the value to calculate the probability of

@return:
    prob: The probability of the particular value

'''
def interp_probability(hist, xedges, val):
    #first, find the nearest neighbors of val
    prob=0
    if val < min(xedges):
        return 0
    elif val > max(xedges):
        return 0

    if val == min(xedges):
        return hist[0]
    elif val == max(xedges):
        return hist[-1]

    else:
        for idx in range(len(xedges)):
            if xedges[idx]>val: #this edge is larger and the previous edge is the lower bound
                lower = xedges[idx-1]
                upper = xedges[idx]

                prob = np.interp(val, xedges, hist)
    return prob


'''

'''



'''
#now let's load all of the chains from the single psr runs:
datadir='/home/jerry/nanograv/11yr/nano11_bwm_master/single_psr_11yr_psrs/data/'
cores=[]
hists_dict = {}
for psr in psrs:
    core = Core(label=psr.name, chaindir = datadir+psr.name+'/')
    cores.append(core)
    hist, xedges = np.histogram(core.get_param('bwm_log10_A'), density = True)
    hists_dict[psr.name] = (hist, xedges)
'''



'''
This is not necessary for the new way of doing this
nside = 4
npix = hp.nside2npix(nside) #this gets us 768 sky skypixels

Npol = 8
polarizations = np.arange(Npol) * np.pi/Npol

Nts = 100  # number of time bins
TMIN = 53217.0
TMAX = 57387.0
Ts = np.linspace(TMIN, TMAX, Nts+2)[1:-1]  # don't use first/last t0
'''




#we need to loop through sky-locations,polarizations, characteristic strains, burst epochs, and pulsars
'''
This is the way I calculated it.
We're just going to use the enterprise function in enterprise.signals called bwm_delay, which is
in the block below this code

for loc in range(npix): #The sky-location of the burst
    burst_theta,burst_phi = hp.pix2ang(nside, loc)
    for pol in polarizations:
        for t0 in Ts:
            for strain in burst_strains:
                lnlike = 0
                for psr in psrs:
                    #set up the sky location of the pulsar
                    psr_theta = psr.theta
                    psr_phi = psr.phi
                    psr_unit_vec = psr.pos

                    #now we need to calculate the relevant polarization tensor of the burst
                    #this is done in the above function
                    rotated_polarization = get_polarization_mtx(burst_theta, burst_phi, pol)

                    #Now lets product this with the pulsar skylocation
                    #The transpose looks like it's on the wrong pulsar, but the unit vectors come back as row vectors
                    #and we are pretending vectors are columns
                    pi = psr_unit_vec
                    pj = np.transpose(psr_unit_vec)

                    pi_eij = np.matmul(pi, rotated_polarization)
                    pi_eij_pj = np.matmul(pi_eij, pj) #this should now be a scalar

                    #now we need to calculate the 1+omega*p factor
                    skyloc = [np.sin(burst_theta)*np.cos(burst_phi),np.sin(burst_theta)*np.sin(burst_phi),np.cos(burst_phi)]
                    omega_dot_p = np.matmul(skyloc, np.transpose(psr_unit_vec))

                    burst_amplitude = strain * 2 * (1+omega_dot_p)/(pi_eij_pj)

                    #At this point, we need to look up the burst amplitude in our lookup table
                    #for now let's just print it for testing
                    print(psr.name + " would see an amplitude of: " + str(burst_amplitude[0][0]))
'''

def make_lookup_table(psr, noisefile, outdir, sign):
    log10_burst_amplitudes = np.linspace(-18, -12, 60, endpoint=True) #grid points for the burst strain
    burst_amp_spacing = '-18,-12,60'

    log10_rn_amps = np.linspace(-18, -12, 60, endpoint=True) #grid points for the pulsar red noise
    rn_amp_spacing = '-18,-12,60'

    Ngammas = 70
    gmin = 0
    gmax = 7
    gammas = np.linspace(gmin, gmax, Ngammas, endpoint=True) #grid points for gamma'
    gamma_spacing ='0,7,70'




    if not os.path.exists(outdir + psr.name):
        os.mkdir(outdir + psr.name)
    #now we need to make a pta for this pulsar to look up likelihoods for each amplitude we calculate
    ##################
    #####   PTA   ####
    ##################

    tmin = psr.toas.min() / const.day
    tmax = psr.toas.max() / const.day


    U,_ = utils.create_quantization_matrix(psr.toas)
    eps = 9  # clip first and last N observing epochs
    t0min = np.floor(max(U[:,eps] * psr.toas/const.day))
    t0max = np.ceil(max(U[:,-eps] * psr.toas/const.day))

    Ts = np.linspace(t0min, t0max, num=100, endpoint=True)
    time_spacing = '{},{},100'.format(t0min, t0max)

    sign_spacing = '-1,1,2'

    pta = ee_models.model_ramp([psr], LogLikelihood,
                          upper_limit=False, bayesephem=False,
                          Tmin_bwm=t0min, Tmax_bwm=t0max)




    with open(noisefile, 'rb') as nfile:
       setpars = json.load(nfile)

    pta.set_default_params(setpars)

    with open(outdir+'{}/pars.txt'.format(psr.name), 'w+') as f:
        f.write('{}_red_noise_gamma;{}\n{}_red_noise_log10_A;{}\n{};{}\n{};{}\n{};{}'.format(psr.name,gamma_spacing,psr.name,rn_amp_spacing, 'ramp_log10_A',burst_amp_spacing, 'ramp_t0',time_spacing,'sign', sign_spacing))

    with open(outdir + "{}/{}_{}.txt".format(psr.name, psr.name, sign),'a+') as f:
        for t0 in Ts:
            for log10_strain in log10_burst_amplitudes:
                #set up the sky location of the pulsar
                #ramp_amp*=sign
                #print(psr.name + "would see an amplitude of: " + str(ramp_amp))

                #Now we need to add the A_red and gamma_red params so that we have in total:
                #A_red, Gamma_red, A_ramp, t_ramp
                for log10_rn_amp in log10_rn_amps:
                    for gamma in gammas:
                        #now we have the four parameters, we need to ask the pta to calculate a likelihood
                        #the pta params are in the order:
                        #[sign, psr_gamma, psr_log10_A, ramp_log10_A, ramp_t0]
                        lnlike = pta.get_lnlikelihood([gamma, log10_rn_amp, log10_strain, t0, sign])
                        f.write('{0:.12f}\n'.format(float(lnlike)))


if __name__ == '__main__':
    pkl_path = '/home/nima/nanograv/11yr_burst_factorizedlikelihood/NANOGrav_11yr_DE436.pickle'
    outdir = '/home/nima/nanograv/11yr_burst_factorizedlikelihood/single_psr_lookup_v4/'
    noisefile = '/home/nima/nanograv/11yr_burst_factorizedlikelihood/noisefiles/noisedict.json'

    with open(pkl_path, 'rb') as f:
        allpsrs=pickle.load(f)
    psr = allpsrs[0]

    params1=[psr, noisefile, outdir, 1]
    params2=[psr, noisefile, outdir, -1]

    log10_burst_amplitudes = np.linspace(-18, -12, 60, endpoint=True) #grid points for the burst strain
    burst_amp_spacing = '-18,-12,60'

    log10_rn_amps = np.linspace(-18, -12, 60, endpoint=True) #grid points for the pulsar red noise
    rn_amp_spacing = '-18,-12,60'


    gammas = np.linspace(0, 7, 70, endpoint=True) #grid points for gamma'
    gamma_spacing ='0,7,70'

    tmin = psr.toas.min() / const.day
    tmax = psr.toas.max() / const.day


    U,_ = utils.create_quantization_matrix(psr.toas)
    eps = 9  # clip first and last N observing epochs
    t0min = np.floor(max(U[:,eps] * psr.toas/const.day))
    t0max = np.ceil(max(U[:,-eps] * psr.toas/const.day))

    Ts = np.linspace(t0min, t0max, num=100, endpoint=True)
    time_spacing = '{},{},100'.format(t0min, t0max)

    sign_spacing = '-1,1,2'


    with open(outdir+'{}/pars.txt'.format(psr.name), 'w+') as f:
        f.write('{}_red_noise_gamma;{}\n{}_red_noise_log10_A;{}\n{};{}\n{};{}\n{};{}'.format(psr.name,gamma_spacing,psr.name,rn_amp_spacing, 'ramp_log10_A',burst_amp_spacing, 'ramp_t0',time_spacing,'sign', sign_spacing))

    pool = mp.Pool(2)
    pool.starmap(make_lookup_table, [params1, params2])
    #for p in psrs:
    #    params.append([p])

    #pool = mp.Pool(mp.cpu_count())
    #pool.starmap(make_lookup_table, params)
