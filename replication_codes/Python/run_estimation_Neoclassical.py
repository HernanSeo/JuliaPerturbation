#########################################################################################################
# Pckages #
#########################################################################################################
from collections import namedtuple
from sympy import *
import numpy
import scipy
import numpy.random as rand
import time
import threading
# from scipy.stats import norm
# from scipy.stats import beta
# from scipy.stats import gamma
from scipy.stats import invgamma
from scipy.stats import truncnorm
import math

num_threads = 15

# from solution_functions_7_0 import *



#########################################################################################################
# Functions #
#########################################################################################################
model = namedtuple("model", ["parameters", "estimate", "estimation", "np", "ns", "priors",
                            "x", "y", "xp", "yp", "variables", "nx", "ny", "nvar", 
                            "e", "eta", "ne",
                            "f", "nf",
                            "SS", "PAR_SS",
                            "flag_order", "flag_deviation", "flag_SSsolver"])

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
    # start = time.time()
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
        # end = time.time()
        # print(end-start)
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

            # start1 = time.time()
            # nfypyp  =   numpy.reshape(permutedims(fypyp, (1, 0, 2)), (model.nf*model.ny, model.ny), order='F')
            # end1 = time.time()
            # print(end1-start1)
            # start2 = time.time()
            # nfypyp = numpy.empty((model.nf * model.ny, model.ny), order='F')
            # for i in range(model.nf):
            #     for j in range(model.ny):
            #         for k in range(model.ny):
            #             nfypyp[i * model.ny + j, k] = fypyp[i, j, k]
            # end2 = time.time()
            # print(end2-start2)
            # nfypyp  =   numpy.reshape(permutedims(fypyp, (1, 0, 2)), (model.nf*model.ny, model.ny), order='F')
            # nfypy   =   numpy.reshape(permutedims(fypy, (1, 0, 2)), (model.nf*model.ny, model.ny), order='F')
            # nfypxp  =   numpy.reshape(permutedims(fypxp, (1, 0, 2)), (model.nf*model.ny, model.nx), order='F')
            # nfypx   =   numpy.reshape(permutedims(fypx, (1, 0, 2)), (model.nf*model.ny, model.nx), order='F')

            # nfyyp  =   numpy.reshape(permutedims(fyyp, (1, 0, 2)), (model.nf*model.ny, model.ny), order='F')
            # nfyy   =   numpy.reshape(permutedims(fyy, (1, 0, 2)), (model.nf*model.ny, model.ny), order='F')
            # nfyxp  =   numpy.reshape(permutedims(fyxp, (1, 0, 2)), (model.nf*model.ny, model.nx), order='F')
            # nfyx   =   numpy.reshape(permutedims(fyx, (1, 0, 2)), (model.nf*model.ny, model.nx), order='F')

            # nfxpyp  =   numpy.reshape(permutedims(fxpyp, (1, 0, 2)), (model.nf*model.nx, model.ny), order='F')
            # nfxpy   =   numpy.reshape(permutedims(fxpy, (1, 0, 2)), (model.nf*model.nx, model.ny), order='F')
            # nfxpxp  =   numpy.reshape(permutedims(fxpxp, (1, 0, 2)), (model.nf*model.nx, model.nx), order='F')
            # nfxpx   =   numpy.reshape(permutedims(fxpx, (1, 0, 2)), (model.nf*model.nx, model.nx), order='F')

            # nfxyp  =   numpy.reshape(permutedims(fxyp, (1, 0, 2)), (model.nf*model.nx, model.ny), order='F')
            # nfxy   =   numpy.reshape(permutedims(fxy, (1, 0, 2)), (model.nf*model.nx, model.ny), order='F')
            # nfxxp  =   numpy.reshape(permutedims(fxxp, (1, 0, 2)), (model.nf*model.nx, model.nx), order='F')
            # nfxx   =   numpy.reshape(permutedims(fxx, (1, 0, 2)), (model.nf*model.nx, model.nx), order='F')
            # end1 = time.time()
            # print(end1-start1)

            # start2 = time.time()
            nfypyp = numpy.empty((model.nf * model.ny, model.ny), order='F')
            nfypy = numpy.empty((model.nf * model.ny, model.ny), order='F')
            nfypxp = numpy.empty((model.nf * model.ny, model.nx), order='F')
            nfypx = numpy.empty((model.nf * model.ny, model.nx), order='F')

            nfyyp = numpy.empty((model.nf * model.ny, model.ny), order='F')
            nfyy = numpy.empty((model.nf * model.ny, model.ny), order='F')
            nfyxp = numpy.empty((model.nf * model.ny, model.nx), order='F')
            nfyx = numpy.empty((model.nf * model.ny, model.nx), order='F')

            nfxpyp = numpy.empty((model.nf * model.nx, model.ny), order='F')
            nfxpy = numpy.empty((model.nf * model.nx, model.ny), order='F')
            nfxpxp = numpy.empty((model.nf * model.nx, model.nx), order='F')
            nfxpx = numpy.empty((model.nf * model.nx, model.nx), order='F')

            nfxyp = numpy.empty((model.nf * model.nx, model.ny), order='F')
            nfxy = numpy.empty((model.nf * model.nx, model.ny), order='F')
            nfxxp = numpy.empty((model.nf * model.nx, model.nx), order='F')
            nfxx = numpy.empty((model.nf * model.nx, model.nx), order='F')

            # Loop to reshape and permute dimensions
            for i in range(model.nf):
                for j in range(model.ny):
                    for k in range(model.ny):
                        nfypyp[i * model.ny + j, k] = fypyp[i, j, k]
                        nfypy[i * model.ny + j, k] = fypy[i, j, k]
                        nfyyp[i * model.ny + j, k] = fyyp[i, j, k]
                        nfyy[i * model.ny + j, k] = fyy[i, j, k]
                    for k in range(model.nx):
                        nfypxp[i * model.ny + j, k] = fypxp[i, j, k]
                        nfypx[i * model.ny + j, k] = fypx[i, j, k]
                        nfyxp[i * model.ny + j, k] = fyxp[i, j, k]
                        nfyx[i * model.ny + j, k] = fyx[i, j, k]
                for j in range(model.nx):
                    for k in range(model.ny):
                        nfxpyp[i * model.nx + j, k] = fxpyp[i, j, k]
                        nfxpy[i * model.nx + j, k] = fxpy[i, j, k]
                        nfxyp[i * model.nx + j, k] = fxyp[i, j, k]
                        nfxy[i * model.nx + j, k] = fxy[i, j, k]
                    for k in range(model.nx):
                        nfxpxp[i * model.nx + j, k] = fxpxp[i, j, k]
                        nfxpx[i * model.nx + j, k] = fxpx[i, j, k]
                        nfxxp[i * model.nx + j, k] = fxxp[i, j, k]
                        nfxx[i * model.nx + j, k] = fxx[i, j, k]
            # end2 = time.time()
            # print(end2-start2)
            # end = time.time()
            # print(end-start)

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
    
    for i in range(npart):
        p0 = parasim[0,i,:]
        loglh[i] = objfun(data, p0, phi_smc, model, nobs, PAR)
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
        eps = 1e-6
        for t in range(10):
            leftR = numpy.linalg.eig(R)[1]
            tauR = numpy.linalg.eig(R)[0]
            # print(tauR < eps)
            tauR[tauR < eps] = eps
            R = leftR * numpy.diag(tauR) *leftR.T
            # print(R)

        # Rdiag   = numpy.diagonal(R)#diag(diag(R)) # covariance with diag elements
        # Rchol = numpy.linalg.cholesky(R).T
        # Rchol2 = numpy.sqrt(Rdiag)
        estimMean[ii,:] = numpy.repeat(mu,model.ns).T
        
        # Particle mutation [Algorithm 2]
        temp_acpt = numpy.zeros(npart) #initialize accpetance indicator

        
        def mutation_RWMH_distributed(j, ii):
            ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(data, para[j,:], loglh[j], logpost[j], c, R, model.ns, phi_smc, nobs, ii, PAR)
                
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
    # sourcery skip: avoid-builtin-shadow
    string_SS, string_ShockVAR, string_PAR_SS, string_deriv = process_model(model)
    exec(string_ShockVAR)
    exec(string_PAR_SS)
    exec(string_SS)
    exec(string_deriv)

    t0 = time.time()
    # npara = initial_para.shape[0]
    # print(npara)
    phi_bend = true
    nobs = data.shape[1]
    acpt=0.25
    trgt=0.25

    ZZ = numpy.dot(numpy.eye(nobs, ny),sol_mat['gx'])
    HH = numpy.eye(nobs, nobs)*0.0001
    DD = numpy.zeros(nobs)
    At = numpy.zeros((model.ns, 1), dtype=float)

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
    
    for i in range(npart):
        # t2 = time.time()
        p0 = parasim[0,i,:]
        # t3 = time.time()
        loglh[i] = objfun(data, p0, phi_smc, model, nobs, PAR, VAR, PAR_SS, SS, deriv, sol_mat, 
                          ZZ, HH, DD, 
                          At)
        # t4 = time.time()
        prior_val = logpriors(p0)
        # t5 = time.time()
        logpost[i] = loglh[i] + prior_val
        # t6 = time.time()
        # print(t3-t2)
        # print(t4-t3)
        # print(t5-t4)
        # print(t6-t5)
        # time.sleep(10)

    loglh[numpy.isnan(loglh)] = -1e50
    loglh[numpy.isinf(loglh)] = -1e50
    logpost[numpy.isnan(logpost)] = -1e50
    logpost[numpy.isinf(logpost)] = -1e50
    t1 = time.time()
    print(t1-t0)

    # ------------------------------------------------------------------------
    # Recursion: For n=2,...,N_[\phi]
    # ------------------------------------------------------------------------
    estimMean = numpy.zeros((nphi, model.ns))

    print("SMC recursion starts ... ")
    # eps = 1e-6
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
        ESS = 1/sum(wtsim[:, ii]**2)
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
        # for _ in range(10):
        #     leftR = numpy.linalg.eig(R)[1]
        #     tauR = numpy.linalg.eig(R)[0]
        #     # print(tauR < eps)
        #     tauR[tauR < eps] = eps
        #     R = leftR * numpy.diag(tauR) *leftR.T
        # Rdiag   = numpy.diagonal(R)#diag(diag(R)) # covariance with diag elements
        # Rchol = numpy.linalg.cholesky(R).T
        # Rchol2 = numpy.sqrt(Rdiag)
        estimMean[ii,:] = numpy.repeat(mu,model.ns).T

        # Particle mutation [Algorithm 2]
        temp_acpt = numpy.zeros(npart) #initialize accpetance indicator

        # propmode=1

        for j in range(npart): #iteration over particles
            # Mutation with RWMH
            # print(type(para))
            ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(data, para[j,:], loglh[j], logpost[j], c, R, model.ns, phi_smc, nobs, ii, PAR, eta, PAR_SS, SS, deriv, sol_mat,
                                                                    ZZ, HH, DD, 
                                                                    At)
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

def objfun(yy, p0, phi_smc, model, nobs, PAR, eta, PAR_SS, SS, deriv, sol_mat, 
           ZZ, HH, DD, At):
    # t10 = time.time()
    for ii in range(model.ns):
        PAR[model.estimation[ii]] = p0[ii]
    # print(PAR)
    # t11 = time.time()
    VAR = eval_ShockVAR(PAR)
    # t12 = time.time()
    PAR_SS = eval_PAR_SS(PAR)
    # t13 = time.time()
    # print(PAR_SS)
    SS = eval_SS(PAR_SS)
    # t14 = time.time()
    # print(SS)
    deriv = eval_deriv(PAR_SS, SS)
    # t15 = time.time()
    # print(deriv)

    sol_mat = solve_model(model, deriv, VAR)
    # t16 = time.time()

    if sol_mat['qzflag'] == 1:    
        aaa =  kf(data,
                sol_mat['hx'],                                  # TT = hx;
                numpy.dot(VAR,VAR.T),                                                # RQR = VAR;
                DD,                                   # DD = 0.0*ones(nobs,1); mean_obs, constants in the observable equation
                ZZ,     # ZZ = B*gx;
                HH,   # HH = [0.0001 0.0 0.0 ;0.0 0.0001 0.0 ; 0.0 0.0 0.0001]#zeros(3,3)# diag(nVAR_me);   m.e. must be in STD
                At)
        # t17 = time.time()
        # print(t11-t10)
        # print(t12-t11)
        # print(t13-t12)
        # print(t14-t13)
        # print(t15-t14)
        # print(t16-t15)
        # print(t17-t16)
        return aaa
    else:
        return -1000000.0

    # print(loglh[i])

def kf(y, TT, RQR, DD, ZZ, HH, At):
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

def mutation_RWMH(data, p0, l0, lpost0, c, R, npara, phi_smc, nobs, i, PAR, eta, PAR_SS, SS, deriv, sol_mat, ZZ, HH, DD, At):
    
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
        lnpY = objfun(data, px.T, phi_smc, model, nobs, PAR, eta, PAR_SS, SS, deriv, sol_mat,
                      ZZ, HH, DD, 
                      At)
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
flag_order = 1; flag_deviation = true; flag_SSsolver = false

ALPHA, BETA, DELTA, RHO, SIGMA, MUU, AA = symbols('ALPHA BETA DELTA RHO SIGMA MUU AA')
parameters = [ALPHA, BETA, DELTA, RHO, SIGMA, MUU, AA]
estimate    =   [RHO, MUU]
npar = len(parameters)
nsim = len(estimate)
estimation = [3,5]
priors = [priors(0.9, 0.0, 1.0, truncnorm(0.0, 1.0, loc=0.75, scale=0.25)),
            priors(0.05, 0.0, 1e6, invgamma(5.0, 0.25))]
        

k, kp, a, ap, c, cp, n, nnp, yy, yyp, r, rp, ii, iip = symbols('k kp a ap c cp n nnp yy yyp r rp ii iip')
x    =   [k, a]
y    =   [c, n, r, yy, ii]
xp   =   [kp, ap]
yp   =   [cp, nnp, rp, yyp, iip]
variables = [k, a, c, n, r, yy, ii, kp, ap, cp, nnp, rp, yyp, iip]

nx = len(x)
ny = len(y)
nvar = nx + ny

epsilon = symbols('epsilon')
e   =   [epsilon]
eta =   Matrix([0.0, MUU])

ne = len(e)

# Equilibrium conditions
f1  =   c + kp - (1-DELTA) * k - a * pow(k,ALPHA) * pow(n,1-ALPHA)
f2  =   pow(c,-SIGMA) - BETA * pow(cp,-SIGMA) * (ap * ALPHA * pow(kp,ALPHA-1) * pow(nnp,1-ALPHA) + 1 - DELTA)
f3  =   AA - pow(c,-SIGMA) * a * (1-ALPHA) * pow(k,ALPHA) * pow(n,-ALPHA) 
f4  =   log(ap) - RHO * log(a)
f5  =   r - a * ALPHA * pow(k,ALPHA-1) * pow(n,1-ALPHA)
f6  =   yy - a * pow(k,ALPHA) * pow(n,1-ALPHA)
f7  =   ii - (kp - (1-DELTA) * k)

f   =   [f1, f2, f3, f4, f5, f6, f7]

nf = len(f)    

# SS, PAR_SS = build_model_steadystate(ALPHA, BETA, DELTA, RHO, SIGMA, MUU, AA)
A   =   1.0
N   =   2/3
K   =   pow(ALPHA/(1/BETA-1+DELTA),1/(1-ALPHA))*N
C   =   A * pow(K,ALPHA) * pow(N,1-ALPHA) - DELTA*K
R   =   A * ALPHA * pow(K,ALPHA-1.0) * pow(N,1-ALPHA)
YY  =   A * pow(K,ALPHA) * pow(N,1-ALPHA)
II  =   DELTA*K

SS = [
                log(K), # k
                log(A), # a
                log(C), # c
                log(N),
                log(R),
                log(YY),
                log(II),
                log(K), # kp
                log(A), # ap
                log(C), # cp
                log(N),
                log(R),
                log(YY),
                log(II)
            ]
# Parameters to adjust
AA = pow(C,-SIGMA) * (1-ALPHA)*A*pow(K,ALPHA)*pow(N,-ALPHA)
PAR_SS = [ALPHA, BETA, DELTA, RHO, SIGMA, MUU, AA]

model = model(parameters, estimate, estimation, npar, nsim, priors,
                x, y, xp, yp, variables, nx, ny, nvar, 
                e, eta, ne,
                f, nf,
                SS, PAR_SS,
                flag_order, flag_deviation, flag_SSsolver)

string_SS, string_ShockVAR, string_PAR_SS, string_deriv = process_model(model)

exec(string_ShockVAR)
exec(string_PAR_SS)
exec(string_SS)
exec(string_deriv)

# Parametrization
ALPHA  =   0.30
BETA   =   0.95
DELTA  =   1.00
RHO    =   0.90
SIGMA  =   2.00
MUU    =   0.05

PAR     =   [ALPHA, BETA, DELTA, RHO, SIGMA, MUU]

eta = eval_ShockVAR(PAR)
PAR_SS = eval_PAR_SS(PAR)
SS = eval_SS(PAR_SS)
deriv = eval_deriv(PAR_SS, SS)

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
# nsim = 500
simulation_logdev = simulate_model(model, sol_mat, nsim, eta)
data = simulation_logdev[10000:10000+nsim-1,model.nx:model.nx+3]

c       = 0.1
npart   = 500          # of particles
# npart   = 2**13          # of particles
nphi    = 100 # 500         # of stage
lam     = 3#2.1             # bending coeff
start = time.time()

# estimation_results = smc_rwmh_threads(model, data, PAR, eta, PAR_SS, SS, deriv, sol_mat, c, npart, nphi, lam)
estimation_results = smc_rwmh(model, data, PAR, eta, PAR_SS, SS, deriv, sol_mat, c, npart, nphi, lam)
end = time.time()
print(end-start)