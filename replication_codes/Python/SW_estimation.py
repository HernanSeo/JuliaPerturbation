#########################################################################################################
# Pckages #
#########################################################################################################
from collections import namedtuple
from sympy import *
import numpy
import scipy
import numpy.random as rand
import time
# import threading
# from scipy.stats import norm
# from scipy.stats import beta
# from scipy.stats import gamma
from scipy.stats import invgamma
from scipy.stats import truncnorm
import math

# num_threads = 15

# from solution_functions_7_0 import *



#########################################################################################################
# Functions #
#########################################################################################################
model = namedtuple("model", ["parameters", "estimate", "estimation", "np", "ns", "priors",
                            "x", "y", "xp", "yp", "variables", "nx", "ny", "nvar", 
                            "e", "eta", "ne",
                            "f", "nf",
                            "SS", "PAR_SS",
                            "flag_order", "flag_deviation"])

priors = namedtuple("priors", ["iv", "lb", "ub", "d"])

def process_model(model):
    SS_aux = model.SS.copy()
    for ip in range(model.np):
        for ik in range(len(SS_aux)):
            SS_aux[ik] = SS_aux[ik].subs(model.parameters[ip],symbols("PAR["+str(ip)+"]"))

    string_SS = """def eval_SS(PAR): 
                    return""" + str(SS_aux)
    
    eta_aux = model.eta.copy()
    neta = len(eta_aux)
    for i1 in range(eta_aux.shape[0]):
        for i2 in range(eta_aux.shape[1]):
            for ip in range(model.np):
                eta_aux[i1, i2] = eta_aux[i1, i2].subs(model.parameters[ip], symbols("PAR["+str(ip)+"]"))

    string_ShockVAR = """def eval_ShockVAR(PAR):
                            VAR = numpy.zeros((""" + str(neta) + """, """ + str(neta) + """))
                            VAR = """ + str(eta_aux) + """ 
                            return VAR"""

    PAR_SS_aux = model.PAR_SS.copy()
    for i in range(len(PAR_SS_aux)):
        for ip in range(model.np):
            PAR_SS_aux[i] = PAR_SS_aux[i].subs(model.parameters[ip], symbols("PAR["+str(ip)+"]"))

    string_PAR_SS = """def eval_PAR_SS(PAR):
                        return """ + str(PAR_SS_aux)
    

    f_aux = model.f.copy()
    if model.flag_deviation:
        for j in model.variables:
            for k in range(model.nf):
                f_aux[k] = f_aux[k].subs(j, exp(j))

    fd_v = []
    fd_i = []
    fd_j = []
    if model.flag_order > 1:
        sd_v = []
        sd_i = []
        sd_j = []
        sd_k = []
        if model.flag_order > 2:
            td_v = []
            td_ci = []

    for i in range(model.nvar):
        for j in range(2*(model.nx+model.ny)):
            fd_aux = diff(f_aux[i], model.variables[j])
            if fd_aux != 0:
                fd_v.append(fd_aux)
                fd_i.append(i)
                fd_j.append(j)
                if model.flag_order > 1:
                    for k in range(2*(model.nx+model.ny)):
                        sd_aux = diff(f_aux[i], model.variables[j], model.variables[k])
                        if sd_aux != 0:
                            sd_v.append(sd_aux)
                            sd_i.append(i)
                            sd_j.append(j)
                            sd_k.append(k)
    if model.flag_order == 1:
        for iv in range(2*(model.nx+model.ny)):
            for k in range(len(fd_v)):
                fd_v[k] = fd_v[k].subs(model.variables[iv], symbols("SS[" + str(iv) + "]"))
        for ip in range(model.np):
            for k in range(len(fd_v)):
                fd_v[k] = fd_v[k].subs(model.parameters[ip], symbols("PAR[" + str(ip) + "]"))
        
        string_deriv = """def eval_deriv(PAR, SS): 
                                n = """ + str(model.nvar) + """ 
                                nx = """ + str(model.nx) + """ 
                                ny = """ + str(model.ny) + """
                                fd_i = """ + str(fd_i) + """ 
                                fd_j = """ + str(fd_j) + """ 
                                # deriv = {'d1':[[0 for col in range(2*(nx+ny))] for row in range(n)], 'd2':[], 'd3':[]}
                                deriv = {'d1':numpy.zeros((n,2*(nx+ny))), 'd2':[], 'd3':[]}
                                # print(deriv)
                                d1 = """ + str(fd_v) + """
                                for ii in range(len(d1)): 
                                    deriv['d1'][fd_i[ii]][fd_j[ii]] = d1[ii]
                                return deriv"""
        
    elif model.flag_order == 2:
        for iv in range(2*(model.nx+model.ny)):
            for ik1 in range(len(fd_v)):
                fd_v[ik1] = fd_v[ik1].subs(model.variables[iv], symbols("SS[" + str(iv) + "]"))
            for ik2 in range(len(sd_v)):
                sd_v[ik2] = sd_v[ik2].subs(model.variables[iv], symbols("SS[" + str(iv) + "]"))
        for ip in range(model.np):
            for ik1 in range(len(fd_v)):
                fd_v[ik1] = fd_v[ik1].subs(model.parameters[ip], symbols("PAR[" + str(ip) + "]"))
            for ik2 in range(len(sd_v)):
                sd_v[ik2] = sd_v[ik2].subs(model.parameters[ip], symbols("PAR[" + str(ip) + "]"))
        string_deriv = """def eval_deriv(PAR, SS):
                            n = """ + str(model.nvar) + """ 
                            nx = """ + str(model.nx) + """ 
                            ny = """ + str(model.ny) + """ 
                            fd_i = """ + str(fd_i) + """
                            fd_j = """ + str(fd_j) + """  
                            sd_i = """ + str(sd_i) + """
                            sd_j = """ + str(sd_j) + """
                            sd_k = """ + str(sd_k) + """
                            # deriv = {'d1':[[0 for col in range(2*(nx+ny))] for row in range(n)], 'd2':[[0 for col in range(4*pow(nx+ny,2))] for row in range(n)], 'd3':[]} 
                            deriv = {'d1':numpy.zeros((n,2*(nx+ny))), 'd2':numpy.zeros((n,2*(nx+ny),2*(nx+ny))), 'd3':[]} 
                            d1 = """ + str(fd_v) + """ 
                            for ii in range(""" + str(len(fd_v)) + """):
                                deriv['d1'][(fd_i[ii],fd_j[ii])] = d1[ii] 
                            d2 = """ + str(sd_v) + """
                            for ii in range(""" + str(len(sd_v)) + """):
                                deriv['d2'][(sd_i[ii],sd_j[ii],sd_k[ii])] = d2[ii] 
                            return deriv"""
        
    # print(string_deriv)

    return string_SS, string_ShockVAR, string_PAR_SS, string_deriv

def solve_model(model, deriv, eta):
    nx = model.nx
    ny = model.ny
    nvar = model.nvar

    fd = numpy.array(deriv['d1'])
    sd = numpy.array(deriv['d2'])

    fx  = fd[:,0:nx]
    fy  = fd[:,nx:nvar]
    fxp = fd[:,nvar:nvar+nx]
    fyp = fd[:,nvar+nx:2*nvar]

    A = numpy.concatenate((-fxp, -fyp),axis=1)
    B = numpy.concatenate((fx, fy),axis=1)

    F = scipy.linalg.ordqz(A, B, sort='ouc')

    # Pick non-explosive (stable) eigenvalues
    slt = numpy.abs(numpy.diag(F[1], 0)) < numpy.abs(numpy.diag(F[0], 0))
    nk = numpy.sum(slt)

    # Split up the results appropriately
    z21 = F[5][nk:nvar+1,0:nk]
    z11 = F[5][0:nk,0:nk]

    s11 = F[0][0:nk,0:nk]
    t11 = F[1][0:nk,0:nk]

    indic = 1 # indicates that equilibrium exists and is unique
    
    # Identify cases with no/multiple solutions
    if nk > fx.shape[1]:
        # print("The Equilibrium is Locally Indeterminate")
        indic = 2
    elif nk < fx.shape[1]:
        # print("No Local Equilibrium Exists")
        indic = 0

    if numpy.linalg.matrix_rank(z11) < nk:
        # print("Invertibility condition violated")
        indic = 3

    if indic == 1:
        # Compute the Solution
        z11i = scipy.linalg.inv(z11)
        gx = numpy.real(z21 @ z11i)
        hx = numpy.real(z11 @ scipy.linalg.solve(s11, t11) @ z11i)
        # print(gx)
        # print(hx)
        if model.flag_order > 1:
            fypyp = sd[:,(nx+ny+nx):(nx+ny+nx+ny),(nx+ny+nx):(nx+ny+nx+ny)]
            fypxp = sd[:,(nx+ny+nx):(nx+ny+nx+ny),(nx+ny):(nx+ny+nx)]
            fypy = sd[:,(nx+ny+nx):(nx+ny+nx+ny),(nx):(nx+ny)]
            fypx = sd[:,(nx+ny+nx):(nx+ny+nx+ny),0:(nx)]

            fxpyp = sd[:,(nx+ny):(nx+ny+nx),(nx+ny+nx):(nx+ny+nx+ny)]
            fxpxp = sd[:,(nx+ny):(nx+ny+nx),(nx+ny):(nx+ny+nx)]
            fxpy = sd[:,(nx+ny):(nx+ny+nx),(nx):(nx+ny)]
            fxpx = sd[:,(nx+ny):(nx+ny+nx),0:(nx)]

            fyyp = sd[:,(nx):(nx+ny),(nx+ny+nx):(nx+ny+nx+ny)]
            fyxp = sd[:,(nx):(nx+ny),(nx+ny):(nx+ny+nx)]
            fyy = sd[:,(nx):(nx+ny),(nx):(nx+ny)]
            fyx = sd[:,(nx):(nx+ny),0:(nx)]

            fxyp = sd[:,0:(nx),(nx+ny+nx):(nx+ny+nx+ny)]
            fxxp = sd[:,0:(nx),(nx+ny):(nx+ny+nx)]
            fxy = sd[:,0:(nx),(nx):(nx+ny)]
            fxx = sd[:,0:(nx),0:(nx)]

            nfypyp  =   numpy.reshape(permutedims(fypyp, (1, 0, 2)), (model.nf*model.ny, model.ny), order='F')
            nfypy   =   numpy.reshape(permutedims(fypy, (1, 0, 2)), (model.nf*model.ny, model.ny), order='F')
            nfypxp  =   numpy.reshape(permutedims(fypxp, (1, 0, 2)), (model.nf*model.ny, model.nx), order='F')
            nfypx   =   numpy.reshape(permutedims(fypx, (1, 0, 2)), (model.nf*model.ny, model.nx), order='F')

            nfyyp  =   numpy.reshape(permutedims(fyyp, (1, 0, 2)), (model.nf*model.ny, model.ny), order='F')
            nfyy   =   numpy.reshape(permutedims(fyy, (1, 0, 2)), (model.nf*model.ny, model.ny), order='F')
            nfyxp  =   numpy.reshape(permutedims(fyxp, (1, 0, 2)), (model.nf*model.ny, model.nx), order='F')
            nfyx   =   numpy.reshape(permutedims(fyx, (1, 0, 2)), (model.nf*model.ny, model.nx), order='F')

            nfxpyp  =   numpy.reshape(permutedims(fxpyp, (1, 0, 2)), (model.nf*model.nx, model.ny), order='F')
            nfxpy   =   numpy.reshape(permutedims(fxpy, (1, 0, 2)), (model.nf*model.nx, model.ny), order='F')
            nfxpxp  =   numpy.reshape(permutedims(fxpxp, (1, 0, 2)), (model.nf*model.nx, model.nx), order='F')
            nfxpx   =   numpy.reshape(permutedims(fxpx, (1, 0, 2)), (model.nf*model.nx, model.nx), order='F')

            nfxyp  =   numpy.reshape(permutedims(fxyp, (1, 0, 2)), (model.nf*model.nx, model.ny), order='F')
            nfxy   =   numpy.reshape(permutedims(fxy, (1, 0, 2)), (model.nf*model.nx, model.ny), order='F')
            nfxxp  =   numpy.reshape(permutedims(fxxp, (1, 0, 2)), (model.nf*model.nx, model.nx), order='F')
            nfxx   =   numpy.reshape(permutedims(fxx, (1, 0, 2)), (model.nf*model.nx, model.nx), order='F')

            A = (numpy.dot(numpy.kron(numpy.identity(model.nx+model.ny),numpy.dot(hx.T,gx.T)),(numpy.dot(numpy.dot(nfypyp,gx),hx) + numpy.dot(nfypxp,hx) + numpy.dot(nfypy,gx) + nfypx)) + 
                numpy.dot(numpy.kron(numpy.identity(nx+ny),gx.T),(numpy.dot(numpy.dot(nfyyp,gx),hx) + numpy.dot(nfyxp,hx) + numpy.dot(nfyy,gx) + nfyx )) +    
                numpy.dot(numpy.kron(numpy.identity(nx+ny),hx.T),(numpy.dot(numpy.dot(nfxpyp,gx),hx) + numpy.dot(nfxpxp,hx) + numpy.dot(nfxpy,gx) + nfxpx)) + 
                (numpy.dot(numpy.dot(nfxyp,gx),hx) + numpy.dot(nfxxp,hx) + numpy.dot(nfxy,gx) + nfxx))
            # print(A)    
            B = numpy.kron(fyp, hx.T) 
            # print(B)
            C = numpy.kron(fy, numpy.identity(model.nx))
            # print(C)
            D = numpy.kron(numpy.dot(fyp,gx), numpy.identity(nx))+ numpy.kron(fxp, numpy.identity(model.nx))
            # print(D)

            Qq_1 = numpy.array(-numpy.concatenate((numpy.kron(hx.T,B)+numpy.kron(numpy.identity(model.nx),C),numpy.kron(numpy.identity(model.nx), D)), axis=1), dtype='float64') 
            # print(Qq_1)
            Qq_2 = numpy.array(numpy.reshape(A, numpy.size(A), order='F'), dtype='float64') 
            # print(Qq_2)
            Qq = numpy.linalg.solve(Qq_1, Qq_2.T)
            # print(Qq)

            gxx = numpy.reshape(Qq[0:nx**2*ny],(nx,ny,nx), order='C')
            gxx = numpy.moveaxis(numpy.moveaxis(gxx, 0, -1), 0, 1)
            # print(numpy.moveaxis(numpy.moveaxis(gxx, 0, -1), 0, 1))
            # print(gxx)
            hxx = numpy.reshape(Qq[nx**2*ny:],(nx,nx,nx), order='C')
            hxx = numpy.moveaxis(numpy.moveaxis(hxx, 0, -1), 0, 1)
            # print(hxx)

            Qh = numpy.zeros((model.nf,nx))
            Qg = numpy.zeros((model.nf,ny))
            q = numpy.zeros(model.nf)

            for ii in range(model.nf):
                # First Term
                Qh[ii,:] = numpy.dot(fyp[ii,:],gx)
                # print(Qh[i,:])

                # Second Term
                # q[i] = sum(numpy.dot(numpy.dot(numpy.dot(numpy.dot(fypyp[i,:,:],gx),eta).T,gx),eta))
                q[ii] = numpy.dot(numpy.dot(numpy.dot(numpy.dot(fypyp[ii,:,:],gx),eta).T,gx),eta)
                # print(q[ii])

                # Third Term
                # q[i] = q[i] + sum(diag([(fypxp[i,:,:]*eta)'*gx*eta][:,:]));
                q[ii] = q[ii] + numpy.dot(numpy.dot(numpy.dot(fypxp[ii,:,:],eta).T,gx),eta)
                # print(q[ii])

                # Fourth Term
                # q[i] =  q[i] + sum(diag([( reshape(fyp[i,:]'*reshape(gxx,ny,nx**2),nx,nx)*eta)'*eta][:,:]));
                gxx_reshape = numpy.zeros((gxx.shape[1],2))
                for iid in range(gxx.shape[0]):
                    gxx_reshape = numpy.c_[gxx_reshape, gxx[iid,:,:]]
                gxx_reshape = gxx_reshape[:,2:]
                q[ii] = q[ii] + numpy.dot(numpy.dot(numpy.reshape(numpy.dot(fyp[ii,:],gxx_reshape),(model.nx,model.nx)),eta).T,eta)
                # print(q[ii])

                # Fifth Term
                Qg[ii,:] = fyp[ii,:]
                # print(Qg[i,:])    

                # Sixth Term
                Qg[ii,:] = Qg[ii,:] + fy[ii,:]
                # print(Qg[i,:])    

                # Seventh Term
                Qh[ii,:] = Qh[ii,:] + fxp[ii,:]
                # print(Qh[i,:])    

                # Eighth Term
                # q[i] = q[i] + sum(diag([(fxpyp[i,:,:]*gx*eta)'*eta][:,:]));
                q[ii] = q[ii] + numpy.dot(numpy.dot(numpy.dot(fxpyp[ii,:,:],gx),eta).T,eta)
                # print(q[i])

                # Nineth Term
                q[ii] = q[ii] + numpy.dot(numpy.dot(fxpxp[ii,:,:],eta).T,eta)
                # print(q[i]) 

            Qs = scipy.linalg.solve(-numpy.concatenate((Qg, Qh),axis=1),q)

            gss = Qs[0:ny]
            # print(gss)
            hss = Qs[ny:]
            # print(hss)
            return {'gx':gx, 'hx':hx, 'gxx':gxx, 'hxx':hxx, 'gss':gss, 'hss':hss,
                    'qzflag':indic}
        else:
            return {'gx':gx, 'hx':hx,
                    'qzflag':indic}
    else:
        return {'gx':[], 'hx':[],
                    'qzflag':indic}
    
def simulate_model(model, sol_mat, nsim, eta):
    if model.flag_order > 1:  
        sim_shocks = numpy.random.randn(nsim + 10000, model.ne)
        sim_x_f = numpy.zeros((nsim + 10000, model.nx))
        sim_x_s = numpy.zeros((nsim + 10000, model.nx))
        sim_y = numpy.zeros((nsim + 10000, model.ny))
        for t in range(nsim + 10000 - 1):
            # Construct y(t)
            for i in range(model.ny):
                # print(sol_mat['gxx'][i, :, :])

                sim_y[t, i] = numpy.dot(sol_mat['gx'][i, :], (sim_x_f[t, :] + sim_x_s[t, :])) + 0.5 * numpy.dot(numpy.dot(sim_x_f[t, :].T, sol_mat['gxx'][:, i, :]), sim_x_f[t, :]) + 0.5 * sol_mat['gss'][i]

            if t < nsim + 10000 - 1:
                for i in range(model.nx):
                    sim_x_f[t + 1, i] = numpy.dot(sol_mat['hx'][i, :], sim_x_f[t, :]) + eta[i,:] * sim_shocks[t,:].T
                    sim_x_s[t + 1, i] = numpy.dot(sol_mat['hx'][i, :], sim_x_s[t, :]) + 0.5 * numpy.dot(numpy.dot(sim_x_f[t, :].T, sol_mat['hxx'][:, i, :]), sim_x_f[t, :]) + 0.5 * sol_mat['hss'][i]

        return numpy.concatenate((sim_x_f + sim_x_s, sim_y), axis=1)
    else:
        sim_shocks = numpy.random.randn(nsim + 10000, model.ne)
        sim_x = numpy.zeros((nsim + 10000, model.nx))
        sim_y = numpy.zeros((nsim + 10000, model.ny))
        for t in range(nsim + 10000 - 1):
            # Construct y(t)
            for i in range(model.ny):
                # print(sol_mat['gxx'][i, :, :])

                sim_y[t, i] = numpy.dot(sol_mat['gx'][i, :], sim_x[t, :])

            if t < nsim + 10000 - 1:
                for i in range(model.nx):
                    sim_x[t + 1, i] = numpy.dot(sol_mat['hx'][i, :], sim_x[t, :]) + numpy.dot(eta[i,:],sim_shocks[t,:].T)

        return numpy.concatenate((sim_x, sim_y), axis=1)

def smc_rwmh_threads(model, data, PAR, VAR, PAR_SS, SS, deriv, sol_mat, c, npart, nphi, lam):
    string_SS, string_ShockVAR, string_PAR_SS, string_deriv = process_model(model)
    exec(string_ShockVAR)
    exec(string_PAR_SS)
    exec(string_SS)
    exec(string_deriv)
    # npara = initial_para.shape[0]
    # print(npara)
    phi_bend = true
    nobs = data.shape[1]
    acpt=0.25
    trgt=0.25

    # creating the tempering schedule.
    phi_smc = numpy.arange(0, 1.0, 1.0/nphi)
    if phi_bend:
        phi_smc = numpy.power(phi_smc, lam)

    # ------------------------------------------------------------------------
    # matrices for storing
    # ------------------------------------------------------------------------
    wtsim   = numpy.zeros((npart, nphi))        # weights
    zhat    = numpy.zeros(nphi)             # normalization constant
    nresamp = 0                         # record # of iteration resampled

    csim    = numpy.zeros(nphi) # scale parameter
    ESSsim  = numpy.zeros(nphi) # ESS
    acptsim = numpy.zeros(nphi) # average acceptance rate
    rsmpsim = numpy.zeros(nphi) # 1 if re-sampled

    # SMC algorithm starts here

    # ------------------------------------------------------------------------
    # Initialization: Draws from the prior
    # ------------------------------------------------------------------------
    print("SMC starts ... ")
    # drawing initial values from the prior distributions
    parasim = initial_draw(npart, nphi, model.ns) # parameter draws

    wtsim[:, 0]    =  numpy.ones((npart)) * 1.0/npart      # initial weight is equal weights
    zhat[0]        = sum(wtsim[:,0])

    loglh  = numpy.zeros(npart) #log-likelihood
    logpost = numpy.zeros(npart) #log-posterior

    start = time.time()
    for i in range(npart):
        p0 = parasim[0,i,:]
        loglh[i] = objfun(data, p0, phi_smc, model, nobs, PAR, VAR, PAR_SS, SS, deriv, sol_mat)
        prior_val = logpriors(p0)
        logpost[i] = loglh[i] + prior_val
    end = time.time()    
    print(end-start)

    loglh[numpy.isnan(loglh)] = -1e50
    loglh[numpy.isinf(loglh)] = -1e50
    logpost[numpy.isnan(logpost)] = -1e50
    logpost[numpy.isinf(logpost)] = -1e50

    # ------------------------------------------------------------------------
    # Recursion: For n=2,...,N_[\phi]
    # ------------------------------------------------------------------------
    estimMean = numpy.zeros((nphi, model.ns))

    print("SMC recursion starts ... ")
    # for i in range(2):
    for i in range(nphi-1):
        start = time.time()    

        ii = i + 1
        loglh = loglh.real

        #-----------------------------------
        # (a) Correction
        #-----------------------------------
        # incremental weights
        incwt = numpy.exp((phi_smc[ii]-phi_smc[i])*loglh)
        
        # update weights
        wtsim[:, ii] = wtsim[:, i] * incwt
        zhat[ii] = sum(wtsim[:, ii])
        wtsim[:, ii] = wtsim[:, ii] / zhat[ii]
        
        #-----------------------------------
        # (b) Selection
        #-----------------------------------
        ESS = 1/sum(wtsim[:, ii]**2); # Effective sample size()

        if ESS < npart/2:
            import random 
            id = numpy.zeros(npart, dtype=int)
            for iii in range(npart):   
                id[iii] = random.sample(range(npart), k=1, counts=numpy.int_(numpy.round(wtsim[:,ii]*1e6,0)).tolist())[0]
                
            parasim[i, :, :] = parasim[i, id, :]
            #changed this to parasim[i-1] instead of i (idea being that since
            #para takes parameters from parasim[i-1], need to update that one
            loglh            = loglh[id]
            logpost          = logpost[id]
            wtsim[:, ii]      = 1.0/npart * numpy.ones(npart)  # resampled weights are equal weights
            nresamp          = nresamp + 1
            rsmpsim[ii]       = 1

        #--------------------------------------------------------
        # (c) Mutuation
        #--------------------------------------------------------
        # Adapting the transition kernel
        c = c*(0.95 + 0.10*exp(16.0*(acpt-trgt))/(1.0 + exp(16.0*(acpt-trgt))))
        
        # Calculate estimates of mean & variance
        para = numpy.array(parasim[i, :, :], dtype=numpy.float64)
        wght =  numpy.reshape(numpy.repeat(wtsim[:, ii],model.ns),(npart, model.ns), order='C')
        # print(wght)
        mu      = sum(sum(para*wght)) # mean()
        z       = (para - mu*numpy.ones((npart, model.ns)))
        R       = numpy.dot((z*wght).T,z)       # covariance
        # eps = 1e-6
        # for t in range(10):
        #     leftR = numpy.linalg.eig(R)[1]
        #     tauR = numpy.linalg.eig(R)[0]
        #     # print(tauR < eps)
        #     tauR[tauR < eps] = eps
        #     R = leftR * numpy.diag(tauR) *leftR.T
        #     # print(R)

        # Rdiag   = numpy.diagonal(R)#diag(diag(R)) # covariance with diag elements
        # Rchol = numpy.linalg.cholesky(R).T
        # Rchol2 = numpy.sqrt(Rdiag)
        estimMean[ii,:] = numpy.repeat(mu,model.ns).T
        
        # Particle mutation [Algorithm 2]
        temp_acpt = numpy.zeros(npart) #initialize accpetance indicator
        
        def mutation_RWMH_distributed(j, ii):
            ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(data, para[j,:], loglh[j], logpost[j], c, R, model.ns, phi_smc, nobs, ii, PAR, VAR, PAR_SS, SS, deriv, sol_mat)
                
            parasim[ii, j, :] = ind_para
            loglh[j] = ind_loglh
            # print(loglh[j])
            logpost[j] = ind_post
            temp_acpt[j] = ind_acpt

        threads = []
        for j in range(npart):
            thread = threading.Thread(target=mutation_RWMH_distributed, args=(j, ii))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # for j in range(npart): #iteration over particles
        #     # Mutation with RWMH
        #     # print(type(para))
        #     ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(data, para[j,:], loglh[j], logpost[j], c, R, model.ns, phi_smc, nobs, ii, PAR)
        #     # print(ind_loglh)
        #     #ind_loglh[isnan.(ind_loglh)] .= -1e50
        #     #ind_loglh[isinf.(ind_loglh)] .= -1e50    
        #     #ind_logpost[isnan.(ind_logpost)] .= -1e50
        #     #ind_logpost[isinf.(ind_logpost)] .= -1e50
                
        #     parasim[ii, j, :] = ind_para
        #     loglh[j] = ind_loglh
        #     logpost[j] = ind_post
        #     temp_acpt[j] = ind_acpt
        
        acpt = numpy.mean(temp_acpt)
        
        # store
        csim[ii] = c  # Scale parameter
        ESSsim[ii] = ESS  # ESS
        acptsim[ii] = acpt  # Average acceptance rate

        # Print some information
        if ii % 1 == 0:
            phi_smcii = phi_smc[ii]
            print(".............")
            print(f"phi = {phi_smcii}")
            print(f"c = {c}")
            print(f"acpt = {acpt}")
            print(f"ESS = {ESS}, {nresamp}")
            print(".............")

        end = time.time()    
        print(end-start)
    
    # Report summary statistics
    para = parasim[nphi-1, :, :]
    wght = wtsim[:, nphi-1] 

    mu = numpy.sum(para * wght[:, numpy.newaxis], axis=0)
    sig = numpy.sqrt(numpy.sum((para - mu) ** 2 * wght[:, numpy.newaxis], axis=0))

    print("mu:", mu)
    print("sig:", sig)

def smc_rwmh(model, data, PAR, VAR, PAR_SS, SS, deriv, sol_mat, c, npart, nphi, lam):
    string_SS, string_ShockVAR, string_PAR_SS, string_deriv = process_model(model)
    exec(string_ShockVAR)
    exec(string_PAR_SS)
    exec(string_SS)
    exec(string_deriv)
    # npara = initial_para.shape[0]
    # print(npara)
    phi_bend = true
    nobs = data.shape[1]
    nobs0 = data.shape[0]
    acpt=0.25
    trgt=0.25

    ZZ_aux = numpy.eye(nobs, ny)
    HH = numpy.eye(nobs, nobs)*0.0001
    DD = numpy.zeros(nobs)
    At = numpy.zeros((model.nx, 1), dtype=float)
    loglhvec = numpy.empty(nobs0, dtype=float)

    # creating the tempering schedule.
    phi_smc = numpy.arange(0, 1.0, 1.0/nphi)
    if phi_bend:
        phi_smc = numpy.power(phi_smc, lam)

    # ------------------------------------------------------------------------
    # matrices for storing
    # ------------------------------------------------------------------------
    wtsim   = numpy.zeros((npart, nphi))        # weights
    zhat    = numpy.zeros(nphi)             # normalization constant
    nresamp = 0                         # record # of iteration resampled

    csim    = numpy.zeros(nphi) # scale parameter
    ESSsim  = numpy.zeros(nphi) # ESS
    acptsim = numpy.zeros(nphi) # average acceptance rate
    rsmpsim = numpy.zeros(nphi) # 1 if re-sampled

    # SMC algorithm starts here
    t0 = time.time()
    # ------------------------------------------------------------------------
    # Initialization: Draws from the prior
    # ------------------------------------------------------------------------
    print("SMC starts ... ")
    # drawing initial values from the prior distributions
    parasim = initial_draw(npart, nphi, model.ns) # parameter draws

    wtsim[:, 0]    =  numpy.ones((npart)) * 1.0/npart      # initial weight is equal weights
    zhat[0]        = sum(wtsim[:,0])

    loglh  = numpy.zeros(npart) #log-likelihood
    logpost = numpy.zeros(npart) #log-posterior

    for i in range(npart):
        p0 = parasim[0,i,:]
        loglh[i] = objfun(data, p0, phi_smc, model, nobs, PAR, VAR, PAR_SS, SS, deriv, sol_mat, ZZ_aux, HH, DD, At, loglhvec)
        prior_val = logpriors(p0)
        logpost[i] = loglh[i] + prior_val
        
    loglh[numpy.isnan(loglh)] = -1e50
    loglh[numpy.isinf(loglh)] = -1e50
    logpost[numpy.isnan(logpost)] = -1e50
    logpost[numpy.isinf(logpost)] = -1e50

    # ------------------------------------------------------------------------
    # Recursion: For n=2,...,N_[\phi]
    # ------------------------------------------------------------------------
    estimMean = numpy.zeros((nphi, model.ns))
    t1 = time.time()
    print(t1-t0)

    print("SMC recursion starts ... ")
    # for i in range(2):
    for i in range(nphi-1):
        t0 = time.time()
        
        ii = i + 1
        loglh = loglh.real

        #-----------------------------------
        # (a) Correction
        #-----------------------------------
        # incremental weights
        incwt = numpy.exp((phi_smc[ii]-phi_smc[i])*loglh)
        
        # update weights
        wtsim[:, ii] = wtsim[:, i] * incwt
        zhat[ii] = sum(wtsim[:, ii])
        wtsim[:, ii] = wtsim[:, ii] / zhat[ii]
        
        #-----------------------------------
        # (b) Selection
        #-----------------------------------
        ESS = 1/sum(wtsim[:, ii]**2); # Effective sample size()

        if ESS < npart/2:
            import random 
            id = numpy.zeros(npart, dtype=int)
            for iii in range(npart):   
                id[iii] = random.sample(range(npart), k=1, counts=numpy.int_(numpy.round(wtsim[:,ii]*1e6,0)).tolist())[0]
                
            parasim[i, :, :] = parasim[i, id, :]
            #changed this to parasim[i-1] instead of i (idea being that since
            #para takes parameters from parasim[i-1], need to update that one
            loglh            = loglh[id]
            logpost          = logpost[id]
            wtsim[:, ii]      = 1.0/npart * numpy.ones(npart)  # resampled weights are equal weights
            nresamp          = nresamp + 1
            rsmpsim[ii]       = 1

        #--------------------------------------------------------
        # (c) Mutuation
        #--------------------------------------------------------
        # Adapting the transition kernel
        c = c*(0.95 + 0.10*exp(16.0*(acpt-trgt))/(1.0 + exp(16.0*(acpt-trgt))))
        
        # Calculate estimates of mean & variance
        para = numpy.array(parasim[i, :, :], dtype=numpy.float64)
        wght =  numpy.reshape(numpy.repeat(wtsim[:, ii],model.ns),(npart, model.ns), order='C')
        # print(wght)
        mu      = sum(sum(para*wght)) # mean()
        z       = (para - mu*numpy.ones((npart, model.ns)))
        R       = numpy.dot((z*wght).T,z)       # covariance
        eps = 1e-6
        # for t in range(10):
        #     leftR = numpy.linalg.eig(R)[1]
        #     tauR = numpy.linalg.eig(R)[0]
        #     # print(tauR < eps)
        #     tauR[tauR < eps] = eps
        #     R = leftR * numpy.diag(tauR) *leftR.T
        #     # print(R)

        # Rdiag   = numpy.diagonal(R)#diag(diag(R)) # covariance with diag elements
        # Rchol = numpy.linalg.cholesky(R).T
        # Rchol2 = numpy.sqrt(Rdiag)
        estimMean[ii,:] = numpy.repeat(mu,model.ns).T
        
        # Particle mutation [Algorithm 2]
        temp_acpt = numpy.zeros(npart) #initialize accpetance indicator
        
        for j in range(npart): #iteration over particles
            # Mutation with RWMH
            # print(type(para))
            ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(data, para[j,:], loglh[j], logpost[j], c, R, model.ns, phi_smc, nobs, ii, PAR, VAR, PAR_SS, SS, deriv, sol_mat, ZZ_aux, HH, DD, At, loglhvec)
            # print(ind_loglh)
            #ind_loglh[isnan.(ind_loglh)] .= -1e50
            #ind_loglh[isinf.(ind_loglh)] .= -1e50    
            #ind_logpost[isnan.(ind_logpost)] .= -1e50
            #ind_logpost[isinf.(ind_logpost)] .= -1e50
                
            parasim[ii, j, :] = ind_para
            loglh[j] = ind_loglh
            logpost[j] = ind_post
            temp_acpt[j] = ind_acpt

        acpt = numpy.mean(temp_acpt)
        
        # store
        csim[ii] = c  # Scale parameter
        ESSsim[ii] = ESS  # ESS
        acptsim[ii] = acpt  # Average acceptance rate
        t1 = time.time()
        print(t1-t0)
        
        # Print some information
        if ii % 1 == 0:
            phi_smcii = phi_smc[ii]
            print(".............")
            print(f"phi = {phi_smcii}")
            print(f"c = {c}")
            print(f"acpt = {acpt}")
            print(f"ESS = {ESS}, {nresamp}")
            print(".............")

    # Report summary statistics
    para = parasim[nphi-1, :, :]
    wght = wtsim[:, nphi-1] 

    mu = numpy.sum(para * wght[:, numpy.newaxis], axis=0)
    sig = numpy.sqrt(numpy.sum((para - mu) ** 2 * wght[:, numpy.newaxis], axis=0))

    print("mu:", mu)
    print("sig:", sig)

def objfun(yy, p0, phi_smc, model, nobs, PAR, VAR, PAR_SS, SS, deriv, sol_mat,
           ZZ_aux, HH, DD, At, loglhvec):
    for ii in range(model.ns):
        PAR[model.estimation[ii]] = p0[ii]
    # print(PAR)
    VAR = eval_ShockVAR(PAR)
    PAR_SS = eval_PAR_SS(PAR)
    # print(PAR_SS)
    SS = eval_SS(PAR_SS)
    # print(SS)
    deriv = eval_deriv(PAR_SS, SS)
    # print(deriv)
    
    sol_mat = solve_model(model, deriv, VAR)
    
    if sol_mat['qzflag'] == 1:    
        ZZ = numpy.dot(ZZ_aux,sol_mat['gx'])        
        return  kf(data,
                sol_mat['hx'],                                  # TT = hx;
                numpy.dot(VAR,VAR.T),                                                # RQR = VAR;
                DD,                                   # DD = 0.0*ones(nobs,1); mean_obs, constants in the observable equation
                ZZ,     # ZZ = B*gx;
                HH,   # HH = [0.0001 0.0 0.0 ;0.0 0.0001 0.0 ; 0.0 0.0 0.0001]#zeros(3,3)# diag(nVAR_me);   m.e. must be in STD
                At, loglhvec)
        
    else:
        return -1000000.0

    # print(loglh[i])
# def kf(y, TT, RQR, DD, ZZ, HH, t0):
    # print("ZZ shape:", ZZ.shape)
    # print("HH shape:", HH.shape)
    # print("ZZ dtype:", ZZ.dtype)
    # print("HH dtype:", HH.dtype)

    # nobs, ny = y.shape
    # ns, _ = TT.shape

    # At = numpy.zeros((ns, 1), dtype=float)
    # TT_old = TT.copy()
    # RQR_old = RQR.copy()
    # P_10_old = numpy.eye(ns)
    # loglhvec = numpy.empty(nobs, dtype=float)
    # P_10 = P_10_old.copy()
    # diferenz = 0.1
    
    # while diferenz > 1e-25:
    #     P_10 = numpy.dot(numpy.dot(TT_old, P_10_old), TT_old.T) + RQR_old
    #     diferenz = numpy.max(numpy.abs(P_10 - P_10_old))
    #     RQR_old = numpy.dot(numpy.dot(TT_old, RQR_old), TT_old.T) + RQR_old
    #     TT_old = numpy.dot(TT_old, TT_old)
    #     P_10_old = P_10.copy()
    
    # Pt = P_10
    # loglh = 0.0
    # yaux = numpy.empty_like(DD)
    # ZPZ = numpy.empty_like(HH)
    # TTPt = numpy.empty_like(Pt)
    # TAt = numpy.empty_like(At)
    # KiF = numpy.empty_like(At)
    # PtZZ = numpy.dot(Pt, ZZ.T)
    # Kt = numpy.empty_like(PtZZ)
    # TTPtTT = numpy.empty_like(Pt)
    # KFK = numpy.empty_like(Pt)
    # iFtnut = numpy.empty_like(DD)

    # if numpy.isnan(numpy.sum(At)):
    #     print(At)

    # for i in range(nobs):
    #     yaux = numpy.dot(ZZ, At)
    #     yhat = yaux + DD
    #     if numpy.isnan(float(numpy.sum(yhat))):
    #         print(i)
        
    #     nut = (y[i, :] - numpy.array(yhat).T).T
    #     # print("Shape:", Pt.shape)
    #     # print("Dtype:", Pt.dtype)
    #     Pt = numpy.array(Pt, dtype=numpy.float64)
    #     PtZZ = numpy.dot(Pt, ZZ.T)
    #     ZPZ = numpy.dot(ZZ, PtZZ)

    #     Ft = ZPZ + HH
    #     Ft = 0.5 * (Ft + Ft.T)

    #     dFt = numpy.linalg.det(Ft)
    #     iFt = numpy.linalg.inv(Ft)
    #     iFtnut = numpy.dot(iFt, nut)
    #     if numpy.isnan(float(numpy.sum(numpy.dot(nut.T,iFtnut)))):
    #         print(i)
        
    #     loglhvec[i] = -0.5 * numpy.log(dFt) - 0.5 * (numpy.dot(nut.T, iFtnut))[0,0]
    #     TTPt = numpy.dot(TT, Pt)
    #     Kt = numpy.dot(TTPt, ZZ.T)
    #     TAt = numpy.dot(TT, At)
    #     KiF = numpy.dot(Kt, iFtnut)
    #     At = TAt + KiF
    #     TTPtTT = numpy.dot(TTPt, TT.T)
    #     KFK = numpy.dot(Kt, numpy.linalg.solve(Ft, Kt.T))
    #     Pt = TTPtTT - KFK + RQR
    
    # loglh = numpy.sum(loglhvec) - nobs * 0.5 * ny * numpy.log(2 * numpy.pi)
    # return loglh

def kf(y, TT, RQR, DD, ZZ, HH, At, loglhvec):
# def kf(y, TT, RQR, DD, ZZ, HH, At):
    # t21 = time.time()
    nobs, ny = y.shape
    ns, _ = TT.shape
    # t22 = time.time()

    # At = numpy.zeros((ns, 1), dtype=float)
    TT_old = TT.astype(float)
    RQR_old = RQR.astype(float)
    P_10_old = numpy.eye(ns, dtype=float)
    loglhvec = numpy.empty(nobs, dtype=float)
    Pt = numpy.empty_like(P_10_old)
    diferenz = 0.1
    # t23 = time.time()

    while diferenz > 1e-25:
        Pt = numpy.dot(numpy.dot(TT_old, P_10_old), TT_old.T) + RQR_old
        diferenz = numpy.max(numpy.abs(Pt - P_10_old))
        RQR_old = numpy.dot(numpy.dot(TT_old, RQR_old), TT_old.T) + RQR_old
        TT_old = numpy.dot(TT_old, TT_old)
        numpy.copyto(P_10_old, Pt)
    
    # t24 = time.time()

    # Pt = P_10
    loglh = 0.0
    yaux = numpy.empty_like(DD)
    # ZPZ = numpy.empty_like(HH)
    TTPt = numpy.empty_like(Pt)
    # TAt = numpy.empty_like(At)
    # KiF = numpy.empty_like(At)
    # PtZZ = numpy.dot(Pt, ZZ.T)
    # Kt = numpy.dot(Pt, ZZ.T)
    # TTPtTT = numpy.empty_like(Pt)
    # KFK = numpy.empty_like(Pt)
    iFtnut = numpy.empty_like(DD)
    # t25 = time.time()

    # if numpy.isnan(numpy.sum(At)):
    #     print(At)

    # t26 = time.time()

    for i in range(nobs):
        yaux = numpy.dot(ZZ, At)
        yhat = yaux + DD
        # if numpy.isnan(float(numpy.sum(yhat))):
        #     print(i)
        # if numpy.isnan(yhat).any():
        #     print(i)

        # nut = (y[i, :] - numpy.array(yhat).T).T
        nut = y[i, :] - yhat
        # Pt = numpy.array(Pt, dtype=numpy.float64)
        # PtZZ = numpy.dot(Pt, ZZ.T)
        # ZPZ = numpy.dot(ZZ, PtZZ)
        Pt = Pt.astype(numpy.float64)  # Ensure Pt is of dtype np.float64
        # PtZZ = numpy.dot(Pt, ZZ.T)
        # ZPZ = numpy.dot(ZZ, numpy.dot(Pt, ZZ.T))

        # Ft = ZPZ + HH
        Ft = numpy.dot(ZZ, numpy.dot(Pt, ZZ.T)) + HH

        Ft = 0.5 * (Ft + Ft.T)

        dFt = numpy.linalg.det(Ft)
        iFt = numpy.linalg.inv(Ft)
        iFtnut = numpy.dot(iFt, nut)

        # if numpy.isnan(numpy.sum(numpy.dot(nut, iFtnut))):
        #     print(i)
        
        loglhvec[i] = -0.5 * (numpy.log(dFt) + (numpy.dot(nut, iFtnut))[0, 0])
        TTPt = numpy.dot(TT, Pt)
        Kt = numpy.dot(TTPt, ZZ.T)
        # TAt = numpy.dot(TT, At)
        # KiF = numpy.dot(Kt, iFtnut)
        At = numpy.dot(TT, At) + numpy.dot(Kt, iFtnut)
        # TTPtTT = numpy.dot(TTPt, TT.T)
        # KFK = numpy.dot(Kt, numpy.linalg.solve(Ft, Kt.T))
        Pt = numpy.dot(TTPt, TT.T) - numpy.dot(Kt, numpy.linalg.solve(Ft, Kt.T)) + RQR
        # Pt = TTPtTT - KFK + RQR
    # t27 = time.time()

    loglh = numpy.sum(loglhvec) - nobs * 0.5 * ny * numpy.log(2 * numpy.pi)
    # t28 = time.time()

    # print(t22-t21)
    # print(t23-t22)
    # print(t24-t23)
    # print(t25-t24)
    # print(t26-t25)
    # print(t27-t25)
    # print(t28-t27)
    
    return loglh

def initial_draw(npart, nphi, npara):
    parasim = numpy.zeros((nphi, npart, npara)) # parameter draws

    for ii in range(model.ns):
            parasim[0, :, ii] = model.priors[ii].d.rvs(npart)

    return parasim

def logpriors(p0):
    p0 = numpy.array(p0, dtype=float)
    nparams = numpy.size(p0)
    # idx = numpy.nan * numpy.zeros(nparams)
    idx = 0
    # print(p0[0])
    for ii in range(model.ns):
        if p0[ii] < model.priors[ii].lb or p0[ii] > model.priors[ii].ub:
            # idx[ii] = 0.0
            idx = - math.inf
        else:
            aux = model.priors[ii].d.pdf(p0[ii])    
            if aux > 0:
                idx += numpy.log(aux)
            else:
                idx = - math.inf

    return idx

def mutation_RWMH(data, p0, l0, lpost0, c, R, npara, phi_smc, nobs, i, PAR, eta, PAR_SS, SS, deriv, sol_mat, ZZ_aux, HH, DD, At, loglhvec):
    
    # RW proposal
    # print(R)
    
    px = p0 + numpy.dot(c * numpy.linalg.cholesky(R).T, rand.randn(npara))
    # print(px)
    
    prior_val = logpriors(px)
    
    if prior_val == -numpy.inf:
        lnpost = -numpy.inf
        lnpY = -numpy.inf
        lnprio = -numpy.inf
        error = 10
    else:
        lnpY = objfun(data, px.T, phi_smc, model, nobs, PAR, eta, PAR_SS, SS, deriv, sol_mat, ZZ_aux, HH, DD, At, loglhvec)
        lnpost = lnpY + prior_val    
        #lnpost, lnpY, lnprio, error = objfun(data, px, phi_smc, model, nobs)
    
    
        
    # Accept/Reject
    # print(exp(lnpost-lpost0))
    alp = exp(lnpost - lpost0)  # this is RW, so q is canceled out

    if rand.uniform(0, 1) < alp:  # accept
        ind_para = px
        ind_loglh = lnpY
        ind_post = lnpost
        ind_acpt = 1
    else:
        ind_para = p0
        ind_loglh = l0
        ind_post = lpost0
        ind_acpt = 0
    
    return ind_para, ind_loglh, ind_post, ind_acpt


def SS_solver(model, PAR):
    f_aux = model.f.copy()
    for i in range(model.nvar):
        for iv in range(model.nx+model.ny):
            f_aux[i] = f_aux[i].subs(model.variables[model.nx+model.ny+iv], model.variables[iv])
    
    for i in range(model.nvar):
        for ip in range(model.np):
            f_aux[i] = f_aux[i].subs(model.parameters[ip], PAR[ip])
    #     end
    # end
    
    for iif in range(model.nf):
        f_aux[iif] = f_aux[iif].subs(invefl, invef)
        f_aux[iif] = f_aux[iif].subs(cfl, cf)
        f_aux[iif] = f_aux[iif].subs(yfl, yf)
        f_aux[iif] = f_aux[iif].subs(cl, c)
        f_aux[iif] = f_aux[iif].subs(pinfl, pinf)
        f_aux[iif] = f_aux[iif].subs(wl, w)
        f_aux[iif] = f_aux[iif].subs(yl, yy)
        f_aux[iif] = f_aux[iif].subs(rl, r)
        f_aux[iif] = f_aux[iif].subs(a, 0)
        f_aux[iif] = f_aux[iif].subs(b, 0)
        f_aux[iif] = f_aux[iif].subs(qs, 0)
        f_aux[iif] = f_aux[iif].subs(ms, 0)
        f_aux[iif] = f_aux[iif].subs(g, 0)
        f_aux[iif] = f_aux[iif].subs(spinf, 0)
        f_aux[iif] = f_aux[iif].subs(sw, 0)
        f_aux[iif] = f_aux[iif].subs(epinfma, 0)
        f_aux[iif] = f_aux[iif].subs(ewma, 0)
    
    # # solu = solve(f_aux, model.variables[1:model.nx+model.ny])[1]
    solu = solve(f_aux, model.variables[1:model.nx+model.ny])
    # print(solu)
    # solu = nonlinsolve(simplify(f_aux))
    # i = 1
    SS = numpy.zeros(2*(model.nx+model.ny))
    for ii in range(model.nx+model.ny):
        if model.variables[ii] in solu.keys():
            SS[ii] = solu[model.variables[ii]] 
    
    # print(SS)
    return SS


#########################################################################################################
# Estimation #
#########################################################################################################
flag_order = 1; flag_deviation = False; flag_SSsolver = True

ctou, clandaw, cg, curvp, curvw = symbols('ctou clandaw cg curvp curvw')  # fixed parameters
cgamma, cbeta, cpie = symbols('cgamma cbeta cpie') # derived from estimation of: ctrend constebeta constepinf
ctrend, constebeta, constepinf, constelab, calfa, csigma, cfc, cgy = symbols('ctrend constebeta constepinf constelab calfa csigma cfc cgy') # estimated parameters initialisation
csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy = symbols('csadjcost chabb cprobw csigl cprobp cindw cindp czcap crpi crr cry crdy') # estimated parameters initialisation
crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw = symbols('crhoa crhob crhog crhoqs crhoms crhopinf crhow cmap cmaw') # estimated parameters initialisation
clandap, cbetabar, cr, crk, cw, cikbar, cik, clk, cky, ciy, ccy, crkky, cwhlc, cwly = symbols('clandap cbetabar cr crk cw cikbar cik clk cky ciy ccy crkky cwhlc cwly') # derived from steady state
sda, sdb, sdq, sdm, sdg, sdp, sdw = symbols('sda sdb sdq sdm sdg sdp sdw') # std dev of shocks
parameters = [ctou, clandaw, cg, curvp, curvw,
            cgamma, cbeta, cpie, 
            ctrend, constebeta, constepinf, constelab, calfa, csigma, cfc, cgy, 
            csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, 
            crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw,
            clandap, cbetabar, cr, crk, cw, cikbar, cik, clk, cky, ciy, ccy, crkky, cwhlc, cwly,
            sda, sdb, sdq, sdm, sdg, sdp, sdw]
estimate    =   [crhoa, sda]
npar = len(parameters)
nsim = len(estimate)
estimation = [28, 51]
priors = [
        # priors(0.3982, 0.1, 0.8, norm(0.4, 0.10)),
        # priors(0.7420, 0.01, 2.0, gamma(0.25, 0.1)), #_constebeta
        # priors(0.7, 0.1, 2.0, gamma(0.625, 0.1)), #_constepinf
        # priors(1.2918, -10.0, 10.0, norm(0.0, 2.0)), #_constelab
        # priors(0.24, 0.01, 1.0, norm(0.25, 0.05)), #_calfa
        # priors(1.2312, 0.25, 3.0, norm(1.50, 0.375)), #_csigma
        # priors(1.4672, 1.0, 3.0, norm(1.25,0.125)), #_cfc
        # priors(0.05, 0.01, 2.0, norm(0.5, 0.25))] #_cgy
        # prior_csadjcost = (iv = 6.3325, lb = 2, ub = 15, d = Normal(4,1.5)),
        # prior_chabb = (iv = 0.7205, lb = 0.001, ub = 0.99, d = Beta(0.7,0.1)),
        # prior_cprobw = (iv = 0.7937, lb = 0.3, ub = 0.95, d = Beta(0.5,0.1)),
        # prior_csigl = (iv = 2.8401, lb = 0.25, ub = 10, d = Normal(2,0.75)),
        # prior_cprobp = (iv = 0.7813, lb = 0.5, ub = 0.95, d = Beta(0.5,0.10)),
        # prior_cindw = (iv = 0.4425, lb = 0.01, ub = 0.99, d = Beta(0.5,0.15)),
        # prior_cindp = (iv = 0.3291, lb = 0.01, ub = 0.99, d = Beta(0.5,0.15)),
        # prior_czcap = (iv = 0.2648, lb = 0.01, ub = 1, d = Beta(0.5,0.15)),
        # prior_crpi = (iv = 1.7985, lb = 1.0, ub = 3, d = Normal(1.5,0.25)),
        # prior_crr = (iv = 0.8258, lb = 0.5, ub = 0.975, d = Beta(0.75,0.10)),
        # prior_cry = (iv = 0.0893, lb = 0.001, ub = 0.5, d = Normal(0.125,0.05)),
        # prior_crdy = (iv = 0.2239, lb = 0.001, ub = 0.5, d = Normal(0.125,0.05)),
        priors(.9676 , .01, .9999, truncnorm(0.0, 1.0, loc=0.75, scale=0.25)), #_crhoa
        # priors(.2703, .01, .9999, beta(0.5,0.20)), #crhob
        # priors(.9930, .01, .9999, beta(0.5,0.20)), #crhog
        # priors(.5724, .01, .9999, beta(0.5,0.20)), #crhoqs
        # priors(.3, .01, .9999, beta(0.5,0.20)), #crhoms
        # priors(.8692, .01, .9999, beta(0.5,0.20)), #crhopinf
        # priors(.9546, .001, .9999, beta(0.5,0.20)), #crhow
        # priors(.7652, 0.01, .9999, beta(0.5,0.2)), #cmap
        # priors(.8936, 0.01, .9999, beta(0.5,0.2)),    #cmaw
        priors(0.4618, 0.01, 3.0, invgamma(5.0, 0.25)) #sda
        # priors(0.1818513, 0.025, 5, gamma(2.0, 0.1)), #sdb
        # priors(0.6090, 0.01, 3, gamma(2.0, 0.1)), #sdg
        # priors(0.46017, 0.01, 3, gamma(2.0, 0.1)), #sdqs
        # priors(0.2397, 0.01, 3, gamma(2.0, 0.1)), #sdm
        # priors(0.1455, 0.01, 3, gamma(2.0, 0.1)), #sdpinf
        # priors(0.2089, 0.01, 3, gamma(2.0, 0.1))] #sdw
        ]


a, b, qs, ms, g, spinf, sw = symbols('a b qs ms g spinf sw')
epinfma, ewma = symbols('epinfma ewma')
invefl, cfl, yfl = symbols('invefl cfl yfl')
invel, cl, pinfl, wl, yl, rl = symbols('invel cl pinfl wl yl rl')
kpf, kp = symbols('kpf kp')
a_p, b_p, qs_p, ms_p, g_p, spinf_p, sw_p = symbols('a_p b_p qs_p ms_p g_p spinf_p sw_p')
epinfma_p, ewma_p = symbols('epinfma_p ewma_p')
invefl_p, cfl_p, yfl_p = symbols('invefl_p cfl_p yfl_p')
invel_p, cl_p, pinfl_p, wl_p, yl_p, rl_p = symbols('invel_p cl_p pinfl_p wl_p yl_p rl_p')
kpf_p, kp_p = symbols('kpf_p kp_p')
rkf, wf, zcapf, labf, kkf, invef, pkf, rrf, yf, cf = symbols('rkf wf zcapf labf kkf invef pkf rrf yf cf')
mc, rk, w, zcap, lab, kk, inve, pk, r, pinf, yy, c = symbols('mc rk w zcap lab kk inve pk r pinf yy c')
rkf_p, wf_p, zcapf_p, labf_p, kkf_p, invef_p, pkf_p, rrf_p, yf_p, cf_p = symbols('rkf_p wf_p zcapf_p labf_p kkf_p invef_p pkf_p rrf_p yf_p cf_p')
mc_p, rk_p, w_p, zcap_p, lab_p, kk_p, inve_p, pk_p, r_p, pinf_p, y_p, c_p = symbols('mc_p rk_p w_p zcap_p lab_p kk_p inve_p pk_p r_p pinf_p y_p c_p')

x = [a, b, qs, ms, g, spinf, sw,
    epinfma, ewma,
    invefl, cfl, yfl,
    invel, cl, pinfl, wl, yl, rl,
    kpf, kp]
y = [rkf, wf, zcapf, labf, kkf, invef, pkf, rrf, yf, cf,
    mc, rk, w, zcap, lab, kk, inve, pk, r, pinf, yy, c]
xp = [a_p, b_p, qs_p, ms_p, g_p, spinf_p, sw_p,
    epinfma_p, ewma_p,    
    invefl_p, cfl_p, yfl_p,
    invel_p, cl_p, pinfl_p, wl_p, yl_p, rl_p,
    kpf_p, kp_p]
yp = [rkf_p, wf_p, zcapf_p, labf_p, kkf_p, invef_p, pkf_p, rrf_p, yf_p, cf_p,
    mc_p, rk_p, w_p, zcap_p, lab_p, kk_p, inve_p, pk_p, r_p, pinf_p, y_p, c_p]
variables = numpy.concatenate((x,y,xp,yp))
nx = len(x)
ny = len(y)
nvar = nx + ny

# Shock
ea, eb, eqs, ems, eg, espinf, esw = symbols('ea eb eqs ems eg espinf esw')
e = [ea, eb, eqs, ems, eg, espinf, esw]
eta =   Matrix([[sda, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], #a
                [0.0, sdb, 0.0, 0.0, 0.0, 0.0, 0.0],  #b
                [0.0, 0.0, sdq, 0.0, 0.0, 0.0, 0.0],  #qs
                [0.0, 0.0, 0.0, sdm, 0.0, 0.0, 0.0],  #ms
                [cgy, 0.0, 0.0, 0.0, sdg, 0.0, 0.0],  #g
                [0.0, 0.0, 0.0, 0.0, 0.0, sdp, 0.0],  #spinf
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sdw],  #sw
                [0.0, 0.0, 0.0, 0.0, 0.0, sdp, 0.0],  #pinfma
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, sdw],  #wma
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #kpf
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #invefl
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #cfl
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #yfl
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #kp
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #invel
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #cl
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #pinfl
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #wl
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  #yl
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]#rl
                )
ne = len(e)

# Equilibrium conditions
# Exogenous processes
e1 = a_p - crhoa*a # TFP shock
e2 = b_p - crhob*b # Preference shock
e3 = qs_p - crhoqs*qs # Investment shock
e4 = ms_p - crhoms*ms # Monetary shock        
e5 = g_p - crhog*g # Exogenous spending shock
e6 = spinf_p - crhopinf*spinf + cmap*epinfma # Price shock
e7 = sw_p - crhow*sw + cmaw*ewma # Wage shock
e8 = epinfma_p
e9 = ewma_p 

# Lags
l1 = invef - invefl_p
l2 = cf - cfl_p
l3 = inve - invel_p
l4 = yf - yfl_p
l5 = c - cl_p
l6 = pinf - pinfl_p
l7 = w - wl_p
l8 = yy - yl_p
l9 = r - rl_p

# Update of endogenous states
u1 = (1.0-cikbar)*kpf + (cikbar)*invef + (cikbar)*(cgamma**2.0*csadjcost)*qs - kpf_p
u2 = (1.0-cikbar)*kp + cikbar*inve + cikbar*cgamma**2.0*csadjcost*qs - kp_p

# Flexible economy
f1 = calfa * rkf + (1.0-calfa)*wf - 0.0*(1.0-calfa)*a - 1.0*a
f2 = (1.0/(czcap/(1.0-czcap)))*rkf - zcapf
f3 = wf + labf - kkf - rkf
f4 = kpf + zcapf - kkf
f5 = (1.0/(1.0 + cbetabar * cgamma))*(invefl + cbetabar*cgamma*invef_p + (1.0/(cgamma**2.0*csadjcost)) * pkf) + qs - invef
f6 = -rrf - 0.0*b + (1.0/((1.0-chabb/cgamma)/(csigma*(1+chabb/cgamma))))*b + (crk/(crk+(1-ctou)))*rkf_p + ((1.0-ctou)/(crk+(1.0-ctou)))*pkf_p - pkf
f7 = (chabb/cgamma)/(1.0+chabb/cgamma)*cfl + (1.0/(1.0+chabb/cgamma))*cf_p + ((csigma-1.0)*cwhlc/(csigma*(1+chabb/cgamma)))*(labf - labf_p) - (1.0-chabb/cgamma)/(csigma*(1+chabb/cgamma))*(rrf + 0.0*b) + b - cf
f8 = ccy*cf + ciy*invef + g + crkky*zcapf - yf
f9 = cfc*(calfa*kkf + (1.0-calfa)*labf + a) - yf
f10 = csigl*labf + (1.0/(1.0-chabb/cgamma))*cf - (chabb/cgamma)/(1.0-chabb/cgamma)*cfl - wf

# Sticky price - wage economy
s1 = calfa * rk + (1.0 - calfa)*(w) - 1.0*a - 0.0*(1.0-calfa)*a - mc 
s2 = (1.0/(czcap/(1.0-czcap)))*rk - zcap
s3 = w + lab - kk - rk
s4 = kp + zcap - kk
s5 = (1.0/(1.0+cbetabar*cgamma))*(invel + cbetabar*cgamma*inve_p + (1.0/(cgamma**2.0*csadjcost))*pk) + qs - inve
s6 = -r + pinf_p - 0.0*b + (1.0/((1.0-chabb/cgamma)/(csigma*(1.0+chabb/cgamma))))*b + (crk/(crk+(1.0-ctou)))*rk_p + ((1 - ctou)/(crk+(1.0-ctou)))*pk_p - pk
s7 = (chabb/cgamma)/(1.0+chabb/cgamma)*cl + (1.0/(1.0+chabb/cgamma))*c_p + ((csigma-1.0)*cwhlc/(csigma*(1.0+chabb/cgamma)))*(lab - lab_p) - (1.0-chabb/cgamma)/(csigma*(1.0+chabb/cgamma))*(r - pinf_p + 0.0*b) + b - c
s8 = ccy*c + ciy*inve + g + 1.0*crkky*zcap - yy
s9 = cfc*(calfa*kk + (1.0-calfa)*lab + a) - yy
s10 = (1.0/(1.0+cbetabar*cgamma*cindp))*(cbetabar*cgamma*pinf_p + cindp*pinfl + ((1.0-cprobp)*(1.0-cbetabar*cgamma*cprobp)/cprobp)/((cfc - 1.0)*curvp+1.0)*mc) + spinf - pinf
s11 = (1.0/(1.0 + cbetabar*cgamma))*wl + (cbetabar*cgamma/(1.0+cbetabar*cgamma))*w_p + (cindw/(1.0+cbetabar*cgamma))*pinfl - (1.0+cbetabar*cgamma*cindw)/(1.0+cbetabar*cgamma)*pinf + (cbetabar*cgamma)/(1.0+cbetabar*cgamma)*pinf_p + (1.0-cprobw)*(1.0-cbetabar*cgamma*cprobw)/((1.0+cbetabar*cgamma)*cprobw)*(1.0/((clandaw-1.0)*curvw+1.0))*(csigl*lab + (1.0/(1.0-chabb/cgamma))*c - ((chabb/cgamma)/(1.0-chabb/cgamma))*cl - w) + 1.0*sw - w

# Monetary policy rule
m1 = crpi*(1.0 - crr)*pinf + cry*(1.0-crr)*(yy - yf) + crdy*(yy - yf - yl + yfl) + crr*rl + ms - r
    
f = [e1, e2, e3, e4, e5, e6, e7, e8, e9,
    l1, l2, l3, l4, l5, l6, l7, l8, l9,
    u1, u2,
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11,
    m1]
nf = len(f)    

PAR_SS  =   [ctou,         # depreciation rate
            clandaw,      # SS markup labor market
            cg,           # exogenous spending GDP-ratio
            curvp,        # curvature Kimball aggregator goods market
            curvw,        # curvature Kimball aggregator labor market
            ctrend / 100 + 1,
            100 / (constebeta + 100), # discount factor
            constepinf / 100 + 1,
            ctrend,       # quarterly trend growth rate to GDP
            constebeta,
            constepinf,   # quarterly SS inflation rate
            constelab,
            calfa,        # labor share in production
            csigma,       # intertemporal elasticity of substitution
            cfc, 
            cgy,
            csadjcost,                   # investment adjustment cost
            chabb,                   # habit persistence 
            cprobw,                   # calvo parameter labor market
            csigl, 
            cprobp,                   # calvo parameter goods market
            cindw,                   # indexation labor market
            cindp,                   # indexation goods market
            czcap,                   # capital utilization
            crpi,                   # Taylor rule reaction to inflation
            crr,                   # Taylor rule interest rate smoothing
            cry,                   # Taylor rule long run reaction to output gap
            crdy,                   # Taylor rule short run reaction to output gap
            crhoa,
            crhob,
            crhog,
            crhoqs,
            crhoms, 
            crhopinf,
            crhow,
            cmap,
            cmaw,
            cfc,
            cbeta * cgamma**(-csigma),
            cpie / (cbeta * cgamma**(-csigma)),
            (cbeta**(-1.0)) * (cgamma**csigma) - (1.0 - ctou),
            (calfa**calfa * (1.0 - calfa)**(1.0 - calfa) / (clandap * crk**calfa))**(1.0 / (1.0 - calfa)),
            (1.0 - (1.0 - ctou) / cgamma),
            (1.0 - (1.0 - ctou) / cgamma) * cgamma,
            ((1.0 - calfa) / calfa) * (crk / cw),
            cfc * (clk)**(calfa - 1.0),
            cik * cky,
            1.0 - cg - cik * cky,
            crk * cky,
            (1.0 / clandaw) * (1.0 - calfa) / calfa * crk * cky / ccy,
            1.0 - crk * cky,
            sda,
            sdb,
            sdg,
            sdq,
            sdm,
            sdp,
            sdw
            ]
SS = []
# PAR_SS = Array{Sym}([])
# print(eta)

model = model(parameters, estimate, estimation,
            npar, nsim, priors,
            x, y, xp, yp, variables,
            nx, ny, nvar, 
            e, eta,
            ne,
            f,
            nf,
            SS, PAR_SS,
            flag_order, flag_deviation)

string_SS, string_ShockVAR, string_PAR_SS, string_deriv = process_model(model)
exec(string_ShockVAR)
exec(string_PAR_SS)
exec(string_SS)
exec(string_deriv)

# Parametrization
ctou        = 0.025                     # depreciation rate
clandaw     = 1.5                      # SS markup labor market
cg          = 0.18                     # exogenous spending GDP-ratio
curvp       = 10                       # curvature Kimball aggregator goods market
curvw       = 10                       # curvature Kimball aggregator labor market
ctrend      = 0.4312                   # quarterly trend growth rate to GDP
cgamma      = ctrend / 100 + 1
constebeta  = 0.1657
cbeta       = 100 / (constebeta + 100) # discount factor
constepinf  = 0.7869                   # quarterly SS inflation rate
cpie        = constepinf / 100 + 1
constelab   = 0.5509
calfa       = 0.1901                   # labor share in production
csigma      = 1.3808                   # intertemporal elasticity of substitution
cfc         = 1.6064 
cgy         = 0.5187
csadjcost   = 5.7606                   # investment adjustment cost
chabb       = 0.7133                   # habit persistence 
cprobw      = 0.7061                   # calvo parameter labor market
csigl       = 1.8383 
cprobp      = 0.6523                   # calvo parameter goods market
cindw       = 0.5845                   # indexation labor market
cindp       = 0.2432                   # indexation goods market
czcap       = 0.5462                   # capital utilization
crpi        = 2.0443                   # Taylor rule reaction to inflation
crr         = 0.8103                   # Taylor rule interest rate smoothing
cry         = 0.0882                   # Taylor rule long run reaction to output gap
crdy        = 0.2247                   # Taylor rule short run reaction to output gap
crhoa       = 0.9577
crhob       = 0.2194
crhog       = 0.9767
crhoqs      = 0.7113
crhoms      = 0.1479 
crhopinf    = 0.8895
crhow       = 0.9688
cmap        = 0.7010
cmaw        = 0.8503
clandap     = cfc
cbetabar    = cbeta * cgamma**(-csigma)
cr          = cpie / (cbeta * cgamma**(-csigma))
crk         = (cbeta**(-1.0)) * (cgamma**csigma) - (1.0 - ctou)
cw          = (calfa**calfa * (1.0 - calfa)**(1.0 - calfa) / (clandap * crk**calfa))**(1.0 / (1.0 - calfa))
cikbar      = (1.0 - (1.0 - ctou) / cgamma)
cik         = (1.0 - (1.0 - ctou) / cgamma) * cgamma
clk         = ((1.0 - calfa) / calfa) * (crk / cw)
cky         = cfc * (clk)**(calfa - 1.0)
ciy         = cik * cky
ccy         = 1.0 - cg - cik * cky
crkky       = crk * cky
cwhlc       = (1.0 / clandaw) * (1.0 - calfa) / calfa * crk * cky / ccy
cwly        = 1.0 - crk * cky
sda         = 0.4618
sdb         = 1.8513
sdg         = 0.6090
sdq         = 0.6017
sdm         = 0.2397
sdp         = 0.1455
sdw         = 0.2089

PAR = [ctou, clandaw, cg, curvp, curvw,
        cgamma, cbeta, cpie, 
        ctrend, constebeta, constepinf, constelab,
        calfa, csigma, cfc, cgy, 
        csadjcost, chabb, cprobw, csigl, cprobp, cindw, cindp, czcap, crpi, crr, cry, crdy, 
        crhoa, crhob, crhog, crhoqs, crhoms, crhopinf, crhow, cmap, cmaw,
        clandap, cbetabar, cr, crk, cw, cikbar, cik, clk, cky, ciy, ccy, crkky, cwhlc, cwly,
        sda, sdb, sdq, sdm, sdg, sdp, sdw]

eta     =   eval_ShockVAR(PAR)
PAR_SS  =   eval_PAR_SS(PAR)
# SS      =   eval_SS(PAR_SS)
SS      =   SS_solver(model, PAR)
deriv   =   eval_deriv(PAR_SS, SS)

# start = time.time()
# for i in range(100):
#     sol_mat = solve_model(model, deriv, eta)
# end = time.time()
# timing = (end-start)/100
# print(timing)

# start = time.time()
sol_mat = solve_model(model, deriv, eta)
# end =  time.time()
# print(end-start)

# import timeit
# print(timeit.timeit("solve_model(model, deriv, eta)", "from __main__ import solve_model, model, deriv, eta",
#                   number=100)/100)


nsim = 200
# nsim = 200
simulation_logdev = simulate_model(model, sol_mat, nsim, eta)
data = simulation_logdev[10000:10000+nsim-1,model.nx:model.nx+3]
# print(data)

c       = 0.1
npart   = 500          # of particles
# npart   = 2**13          # of particles
nphi    = 100 # 500         # of stage
lam     = 3#2.1             # b ending coeff
# start = time.time()
# estimation_results = smc_rwmh_threads(model, data, PAR, eta, PAR_SS, SS, deriv, sol_mat, c, npart, nphi, lam)
estimation_results = smc_rwmh(model, data, PAR, eta, PAR_SS, SS, deriv, sol_mat, c, npart, nphi, lam)
# end = time.time()
# print(end-start)
