using Random, Distributions, StatsBase #PyPlot,

function smc_rwmh_neoclassical_4_0_v2(initial_para, data, model, PAR, c)
# Code translated from:
# Description: A simple SMC example using the
#   the bimodal "DSGE" model from Schorfheide [2012].
#   The transition kernel is simplified to a random-walk MH.
#
#  For more; see:
#    "Sequential Monte Carlo Sampling for DSGE Models"
#      by Ed Herbst & Frank Schorfheide.
#
# Author: Ed Herbst [edward.p.herbst@frb.gov]
# Last-Updated: 05/09/14
# extended to include the features of the Fortran code by Hernan Seoane

# settings

@unpack flag_order, flag_deviation, flag_SSestimation, parameters, x, y, xp, yp, variables = model
@unpack e, eta, f, n, nx, ny, ne, np, SS, skew = model

npara=size(initial_para,1)
#do_geweke = false
phi_bend = true
npart  = 500#2^13              # of particles
nphi   = 100 # 500         # of stage
lam    = 3#2.1                # bending coeff
nobs = size(data,2)
println(nobs)
acpt=0.25
trgt=0.25

# creating the tempering schedule.
phi_smc = 0:1.0/nphi:1.0
if phi_bend
    phi_smc = phi_smc.^lam
end

#f = (x1, x2) -> objfcn_smc[inputs, x1, x2]; # x1: para, x2: tempering parameter

# ------------------------------------------------------------------------
# matrices for storing
# ------------------------------------------------------------------------
wtsim   = zeros(npart, nphi)        # weights
zhat    = zeros(nphi,1)             # normalization constant
nresamp = 0                         # record # of iteration resampled

csim    = zeros(nphi,1) # scale parameter
ESSsim  = zeros(nphi,1) # ESS
acptsim = zeros(nphi,1) # average acceptance rate
rsmpsim = zeros(nphi,1) # 1 if re-sampled

## SMC algorithm starts here

# ------------------------------------------------------------------------
# Initialization: Draws from the prior
# ------------------------------------------------------------------------
println("SMC starts ... ")
# drawing initial values from the prior distributions

parasim = initial_draw(npart, nphi, npara)



#priorsim       = priordraws[tune.npart]
#parasim[1,:,:] = priorsim;          # from prior

wtsim[:, 1]    = 1.0./npart .* ones(npart,1)      # initial weight is equal weights
zhat[1]        = sum(wtsim[:,1])

# Posterior values at prior draws
loglh  = zeros(npart,1) #log-likelihood
logpost = zeros(npart,1) #log-posterior
#par
Threads.@threads for i=1:npart
    #println(i)
    p0 = parasim[1,i,:]
    #println(size(p0))
    loglh[i] = objfun(data, p0, phi_smc, model, nobs)    
    prior_val = logpriors(p0)
    logpost[i]=loglh[i]+prior_val
    #logpost[i], loglh[i], lnprio[i], error[i] = objfun(data, p0, phi_smc, model, nobs)    
end

loglh[isnan.(loglh)] .= -1e50
loglh[isinf.(loglh)] .= -1e50
logpost[isnan.(logpost)] .= -1e50
logpost[isinf.(logpost)] .= -1e50

# ------------------------------------------------------------------------
# Recursion: For n=2,...,N_[\phi]
# ------------------------------------------------------------------------
estimMean = zeros(nphi, npara)

println("SMC recursion starts ... ")
for i=2:1:nphi
    
    #allReal = isreal(loglh)
    loglh = real(loglh)
    #-----------------------------------
    # (a) Correction
    #-----------------------------------
    # incremental weights
    incwt = exp.((phi_smc[i]-phi_smc[i-1])*loglh)
    
    
    # update weights
    wtsim[:, i] = wtsim[:, i-1].*incwt
    zhat[i]     = sum(wtsim[:, i])
    
    # normalize weights
    wtsim[:, i] = wtsim[:, i]/zhat[i]
    
    #-----------------------------------
    # (b) Selection
    #-----------------------------------
    ESS = 1/sum(wtsim[:, i].^2); # Effective sample size()
    if (ESS .< npart/2)
        id=Vector{Int64}(undef, npart)
        Threads.@threads for ii=1:npart   
            id[ii]=sample(Random.GLOBAL_RNG, 1:npart, weights(wtsim[:,i]), true)[[1]][1]
        end
        parasim[i-1, :, :] = parasim[i-1, id, :]
        #changed this to parasim[i-1] instead of i (idea being that since
        #para takes parameters from parasim[i-1], need to update that one
        loglh            = loglh[id]
        logpost          = logpost[id]
        wtsim[:, i]      = 1.0/npart .* ones(npart,1)  # resampled weights are equal weights
        nresamp          = nresamp + 1
        rsmpsim[i]       = 1
        
    end
    
    #--------------------------------------------------------
    # (c) Mutuation
    #--------------------------------------------------------
    # Adapting the transition kernel
    c = c*(0.95 + 0.10*exp(16.0*(acpt-trgt))/(1 + exp(16.0*(acpt-trgt))))
    
    # Calculate estimates of mean & variance
    para = parasim[i-1, :, :]
    
    wght =  wtsim[:, i].*ones(npart,npara)
    
    mu      = sum(para.*wght) # mean()
    z       = (para - mu.*ones(npart, 2))
    R       = (z.*wght)'*z       # covariance
    Rdiag   = Diagonal(R)#diag(diag(R)) # covariance with diag elements
    Rchol   = cholesky(Hermitian(R)).U
    Rchol2  = sqrt(Rdiag)
    
    estimMean[i,:] .= mu
    
    # Particle mutation [Algorithm 2]
    temp_acpt = zeros(npart,1) #initialize accpetance indicator
    
    propmode=1
    #par
    Threads.@threads for j = 1:npart #iteration over particles
        # Options for proposals
        if propmode .== 1   
            # Mutation with RWMH
            ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(data, para[j,:]', loglh[j], logpost[j], c, R, npara, phi_smc, nobs, i)
            
        elseif propmode .== 2
            # Mutation with Mixture MH
            ind_para, ind_loglh, ind_post, ind_acpt = mutation_MixtureMH(para[j,:]', loglh[j], logpost[j], tune, i)
        end
        #println(ind_loglh)
        #ind_loglh[isnan.(ind_loglh)] .= -1e50
        #ind_loglh[isinf.(ind_loglh)] .= -1e50    
        #ind_logpost[isnan.(ind_logpost)] .= -1e50
        #ind_logpost[isinf.(ind_logpost)] .= -1e50
        
        parasim[i,j,:] = ind_para
        loglh[j]       = ind_loglh
        logpost[j]     = ind_post
        temp_acpt[j,1] = ind_acpt
        
    end
    acpt = mean(temp_acpt); # update average acceptance rate
    
    # store
    csim[i,:]    .= c # scale parameter
    ESSsim[i,:]  .= ESS # ESS
    acptsim[i,:] .= acpt # average acceptance rate
    
    # print some information
    if i % 1 .== 0
        phi_smcii=phi_smc[i]
        println(".............")
        println("phi = $phi_smcii")
        println("c = $c")
        println("acpt = $acpt")
        println("ESS = $ESS,  $nresamp")
        println(".............")
        
        #smctime = tic; # re-start clock()
    end
end


# report summary statistics
para = parasim[nphi, :, :]
wght = repeat(wtsim[:, nphi], 1, size(para,2))

mu = sum(para.*wght, dims = 1)
sig = sum((para .- repeat(mu, npart, 1)).^2 .*wght, dims = 1)

sig = sqrt.(sig)

return mu, sig
end


function mutation_RWMH(data, p0, l0, lpost0, c, R, npara, phi_smc, nobs, i)
    # RWMH for mutation step
    # INPUT
    # tune
    # p0 = para[j,:]'
    # l0 = loglh[j]
    # post0 = logpost[j]
    
    # OUTPUT
    # ind_para
    # ind_loglh
    # ind_post
    # ind_acpt
    
    
    # RW proposal
    px = p0 .+ (c*cholesky(Hermitian(R)).U'*randn(npara,1))'
 
    prior_val = logpriors(px)
    if prior_val == -Inf
        lnpost  = -Inf
        lnpY    = -Inf
        lnprio = -Inf
        error   = 10
    else
        lnpY = objfun(data, px',phi_smc, model, nobs)
        lnpost=lnpY+prior_val
        #lnpost, lnpY, lnprio, error = objfun(data, px',phi_smc, model, nobs)
    end    
    # Accept/Reject
    
    alp = exp(lnpost - lpost0); # this is RW, so q is canceled out

    if rand(Float64) < alp # accept
        ind_para   = px
        ind_loglh  = lnpY
        ind_post   = lnpost
        ind_acpt   = 1
    else
        ind_para   = p0
        ind_loglh  = l0
        ind_post   = lpost0
        ind_acpt   = 0
    end
    return ind_para, ind_loglh, ind_post, ind_acpt  
    end



function generate_random_sized_blocks(prob_include, npara, indices)
    ind2=rperm3(npara)
    indices[ind2] = indices
    break_points = zeros(npara+1,1)
    break_points[1] = 1
    j = 2
    for i in 2: npara
        u=rand()
        if (u .< (1 - prob_include))
            break_points[j] = i
            j = j + 1
        end
    end
    break_points[j] = npara
    if break_points[j] .== break_points[j-1]
        nblocks = j-2
    else
        nblocks = j-1
    end
    return break_points, indices, nblocks
end



function generate_random_blocks(npara, nblocks, indices)
    ind2=rperm3(npara)
    indices[ind2] = indices
    break_points = zeros(npara+1,1)
    gap = floor(npara / nblocks) # check why int & not floor()
    for i in 1: nblocks
       break_points[i] = (i-1)*gap + 1
    end
    break_points[1] = 0
    break_points[nblocks+1] = npara
    return npara, break_points
end

function rperm3(N)
    p = zeros(N,1)
    for j in 1:N
        u = rand()
        k = floor(j*u) + 1
        p[j] = p[k]
        p[k] = j
    end
end

function multinomial_resampling( w , np)
    w = w'
    cw = cumsum(w)
    uu = rand(np, 1)
    indx = Array{Int64, 2}(undef, np, 1)#zeros(np, 1)
    for i in 1:np
        u = uu[i]
        j=1
        while j <= np
           u .< cw[j] ? break : j=j+1
        end
        indx[i] = j
    end
    m = 0

    return indx, m
end


function objfun(yy, p0, phi_smc, model, nobs)
#    function objfun(yy, p0, t0, nobs, deriv_string, SS_string, n, nx, ny)
        t0=1

        for ii in 1:model.ns
            PAR[model.estimation[ii]] = p0[ii]
        end

        VAR     =   eval_ShockVAR(PAR)
        PAR_SS  =   eval_PAR_SS(PAR)
        SS      =   eval_SS(PAR_SS)
        # SS_ER   =   eval_SS_error(PAR_SS, SS)
        deriv   =   eval_deriv(PAR_SS, SS)

        sol_mat     =   solve_perturbation(model, deriv, VAR)

        # println(SS_ER)

        # print(sol_mat.fo_mat.hx)
        if sol_mat.fo_mat.qzflag .== 1
            return kf(yy,
                        sol_mat.fo_mat.hx,                                  # TT = hx;
                        VAR,                                                # RQR = VAR;
                        0.0*ones(nobs,1),                                   # DD = 0.0*ones(nobs,1); mean_obs, constants in the observable equation
                        Matrix{Float64}(I, nobs, model.ny)*sol_mat.fo_mat.gx,     # ZZ = B*gx;
                        [0.00001 0.0 0.0; 0.0 0.00001 0.0; 0.0 0.0 0.00001],   # HH = [0.0001 0.0 0.0 ;0.0 0.0001 0.0 ; 0.0 0.0 0.0001]#zeros(3,3)# diag(nVAR_me);   m.e. must be in STD
                        t0)#, At_ini)
                         
        else
            return -1000000.0
        end
end



# function kf(yy, TT, RQR, DD, ZZ, HH, t0)#, At) # this is a modification to the original code by Herbst where I input directly RQR instead of R & Q
#
#     nobs, ny = size(yy)
#     ns, ~ = size(TT)
#
#     # RQR = RR*QQ*RR'
#
#     At = zeros(ns, 1)
#
#     TT_old=TT
#     RQR_old=RQR
#     P_10_old=Matrix{Float64}(I, size(TT))
#     P_10 =P_10_old
#     diferenz=0.1
#     while diferenz>1e-25
#         P_10 =TT_old*P_10_old*TT_old' + RQR_old
#         diferenz = maximum(abs.(P_10-P_10_old))
#         RQR_old=TT_old*RQR_old*TT_old' + RQR_old
#         TT_old = TT_old * TT_old
#         P_10_old=P_10
#     end    #while diferenz
#     Pt=P_10
#     loglh = 0.0
#     for i in 1:nobs
#         yhat = ZZ*At .+ DD
#         nut = (yy[i, :] .- yhat)'
#         Ft = ZZ*Pt*ZZ' .+ HH
#         Ft = 0.5*(Ft .+ Ft')
#         dFt = det(Ft)
#         iFtnut = Ft \ nut'
#         loglh = loglh - 0.5*ny*log(2*π) - 0.5*log(dFt) - (0.5*nut*iFtnut)[1]
#         TTPt = TT*Pt
#         Kt = TTPt*ZZ'
#         At = TT*At + Kt*iFtnut
#         Pt = TTPt*TT' - Kt*(Ft\Kt') + RQR
#     end
#  return loglh
# end

function kf(y, TT, RQR, DD, ZZ, HH, t0)#, At) # this is a modification to the original code by Herbst where I input directly RQR instead of R & Q

    nobs, ny = size(y)
    ns, ~ = size(TT)

    # RQR = RR*QQ*RR'

    At = Matrix{Float64}(zeros(ns,1))#zeros(ns, 1)

    TT_old=TT
    RQR_old=RQR
    P_10_old=Matrix{Float64}(I, size(TT))
    loglhvec=Array{Float64}(undef, nobs)
    P_10 =P_10_old
    diferenz=0.1
    while diferenz>1e-25
        P_10 =TT_old*P_10_old*TT_old' + RQR_old
        diferenz = maximum(abs.(P_10-P_10_old))
        RQR_old=TT_old*RQR_old*TT_old' + RQR_old
        TT_old = TT_old * TT_old
        P_10_old=P_10
    end    #while diferenz
    Pt=P_10
    loglh = 0.0
    yaux = similar(DD)
    ZPZ = similar(HH)
    TTPt = similar(Pt)
    TAt = similar(At)
    KiF = similar(At)
    PtZZ = similar(Pt*ZZ')
    Kt = similar(PtZZ)
    TTPtTT= similar(Pt)
    KFK= similar(Pt)
    iFtnut = similar(DD)
    if isnan(sum(At))
        println(At)
    end
    for i in 1:nobs
        # yhat = ZZ*At .+ DD
        yhat = mul!(yaux,ZZ,At) + DD
        if isnan(sum(yhat))
            println(i)
        end
        nut = (y[i, :] - yhat)'
        mul!(PtZZ,Pt,ZZ')
        Ft =  mul!(ZPZ,ZZ,PtZZ) + HH #ZZ*Pt*ZZ' .+ HH
        Ft = 0.5*(Ft + Ft')
        dFt = det(Ft)
        mul!(iFtnut,inv(Ft),nut')
        if isnan(sum(nut*iFtnut))
            println(i)
        end
        loglhvec[i] = - 0.5*log(dFt) - (0.5*nut*iFtnut)[1]
        mul!(TTPt,TT,Pt)
        mul!(Kt,TTPt,ZZ')
        At = mul!(TAt,TT,At) + mul!(KiF,Kt,iFtnut)
        Pt = mul!(TTPtTT,TTPt,TT') - mul!(KFK,Kt,(Ft\Kt')) + RQR
    end
    loglh = sum(loglhvec) - nobs*0.5*ny*log(2*π)
 return loglh
end
