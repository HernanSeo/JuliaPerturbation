#########################################################################################################
# Packages #
#########################################################################################################
using SymPy
using LinearAlgebra
using BenchmarkTools
using Parameters
using Distributions
using StatsBase
using Random
# using Plots

#v1.7+
using MKL

#v1.7- 
# BLAS.vendor() 
# :mkl




#########################################################################################################
# Functions #
#########################################################################################################
function process_model(model::NamedTuple; flag_deviation::Bool=true)        
    @unpack parameters, estimate, estimation, npar, ns = model
    @unpack x, y, xp, yp, variables, nx, ny, nvar = model 
    @unpack e, eta, ne = model
    @unpack f, nf = model
    @unpack SS, PAR_SS = model
    @unpack flag_order = model

    eta_aux = eta
    neta = length(eta)
    @inbounds for i1 in 1:size(eta_aux,1)
        @inbounds for i2 in 1:size(eta_aux,2)
            @inbounds for ip in 1:npar
                eta_aux[i1,i2] = subs(eta_aux[i1,i2], parameters[ip], Sym("PAR["*string(ip)*"]"))
            end
        end
    end
    ShockVAR_string = Meta.parse("function eval_ShockVAR(PAR); VAR = Array{Float64}(zeros("*string(neta)*","*string(neta)*")); VAR = "*string(eta_aux)[4:end]*"; return VAR; end")
    eval(ShockVAR_string)
    
    @inbounds for ip in npar
        copyto!(SS, SS.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
    end
    
    SS_string = Meta.parse("function eval_SS(PAR); return " * string(SS)[4:end] * "; end;")
    eval(SS_string)
    
    @inbounds for ip in 1:npar
        for ip2 in 1:npar
            PAR_SS[ip2] = PAR_SS[ip2].subs(parameters[ip],Sym("PAR["*string(ip)*"]"))
        end
    end
    
    PAR_SS_string = Meta.parse("function eval_PAR_SS(PAR); return " * string(PAR_SS)[4:end] * "; end;")
    eval(PAR_SS_string)
    
    if flag_deviation
        @inbounds for j in variables
            # if j ∉ [a, ap]
                copyto!(f,subs.(f, j, (exp(j))))
            # end
        end
    end
    
    # fd_v = Array{Sym}(undef,0); fd_l = Array{Int}(undef,0); fd_c =Array{Int}(undef,0); fd_ci = Array{CartesianIndex}(undef,0)
    fd_v = Array{Sym}(undef,0); fd_ci = Array{CartesianIndex}(undef,0)
    if flag_order > 1
        # sd_v = Array{Sym}(undef,0); sd_l = Array{Int}(undef,0); sd_c =Array{Int}(undef,0); sd_ci = Array{CartesianIndex}(undef,0)
        sd_v = Array{Sym}(undef,0); sd_ci = Array{CartesianIndex}(undef,0)
        if flag_order > 2
            # td_v = Array{Sym}(undef,0); td_l = Array{Int}(undef,0); td_c =Array{Int}(undef,0); td_ci = Array{CartesianIndex}(undef,0)
            td_v = Array{Sym}(undef,0); td_ci = Array{CartesianIndex}(undef,0)
        end
    end

    @inbounds for i in 1:nvar
        @inbounds for j in 1:2*(nx+ny)
            fd_aux = diff(f[i], variables[j])
            if fd_aux != 0
                # for m1 in 1:2*(nx+ny)
                #     fd_aux = subs(fd_aux,variables[m1],SS[m1])
                # end
                push!(fd_v, fd_aux)
                # push!(fd_l, i)
                # push!(fd_c, j)
                push!(fd_ci, CartesianIndex(i,j))
                if flag_order > 1
                    @inbounds for k in 1:2*(nx+ny)
                        sd_aux = diff(f[i], variables[j], variables[k])
                        if sd_aux != 0
                            # for m2 in 1:2*(nx+ny)
                            #     sd_aux = subs(sd_aux,variables[m2],SS[m2])
                            # end
                            push!(sd_v, sd_aux)
                            # push!(sd_l, i)
                            # push!(sd_c, (j-1)*2(nx+ny)+k)
                            # push!(sd_ci, CartesianIndex(i,(j-1)*2(nx+ny)+k))
                            push!(sd_ci, CartesianIndex(i,j,k))
                            if flag_order > 2
                                @inbounds for l in 1:2*(nx+ny)
                                    td_aux = diff(f[i], variables[j], variables[k], variables[l])
                                    if td_aux != 0
                                        # for m3 in 1:2*(nx+ny)
                                        #     td_aux = subs(td_aux,variables[m3],SS[m3])
                                        # end
                                        push!(td_v, td_aux)
                                        # push!(td_l, i)
                                        # push!(td_c, (j-1)*4(nx+ny)^2+(k-1)*2(nx+ny)+l)
                                        # push!(td_ci, CartesianIndex(i,(j-1)*4(nx+ny)^2+(k-1)*2(nx+ny)+l))
                                        push!(td_ci, CartesianIndex(i,j,k,l))
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    if flag_order == 1
        @inbounds for iv in 1:2(nx+ny)
            copyto!(fd_v, fd_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
        end
        @inbounds for ip in 1:npar
            copyto!(fd_v, fd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
        end
        # function_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(nvar) * "; nx = " * string(nx) * "; ny = " * string(ny) * "; fd_l = " * string(fd_l) * "; fd_c = " * string(fd_c) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(0)), d3 = Array{Float64}(zeros(0))); d1 = " * string(fd_v)[4:end] * "; @inbounds for ii in 1:length(d1); deriv.d1[fd_l[ii], fd_c[ii]] = d1[ii]; end; return deriv; end;")
        deriv_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(nvar) * "; nx = " * string(nx) * "; ny = " * string(ny) * "; fd_ci = " * string(fd_ci) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(0)), d3 = Array{Float64}(zeros(0))); d1 = " * string(fd_v)[4:end] * "; @inbounds deriv.d1[fd_ci] = d1; return deriv; end;")    
    elseif flag_order == 2
        @inbounds for iv in 1:2(nx+ny)
            copyto!(fd_v, fd_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
            copyto!(sd_v, sd_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
        end
        @inbounds for ip in 1:npar
            copyto!(fd_v, fd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
            copyto!(sd_v, sd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
        end
        # function_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(nvar) * "; nx = " * string(nx) * "; ny = " * string(ny) * "; fd_l = " * string(fd_l) * "; fd_c = " * string(fd_c) * "; sd_l = " * string(sd_l) * "; sd_c = " * string(sd_c) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(n, 4*(nx+ny)^2)), d3 = Array{Float64}(zeros(0))); d1 = " * string(fd_v)[4:end] * "; for ii in 1:length(d1); deriv.d1[fd_l[ii],fd_c[ii]] = d1[ii]; end; d2 =" * string(sd_v)[4:end] * "; @inbounds for ii in 1:length(d2); deriv.d2[sd_l[ii],sd_c[ii]] = d2[ii]; end; return deriv; end;")
        deriv_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(nvar) * "; nx = " * string(nx) * "; ny = " * string(ny) * "; fd_ci = " * string(fd_ci) * "; sd_ci = " * string(sd_ci) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(n, 2*(nx+ny), 2*(nx+ny))), d3 = Array{Float64}(zeros(0))); d1 = " * string(fd_v)[4:end] * "; @inbounds deriv.d1[fd_ci] = d1; d2 =" * string(sd_v)[4:end] * "; @inbounds deriv.d2[sd_ci] = d2; return deriv; end;")
    elseif flag_order == 3
        @inbounds for iv in 1:2(nx+ny)
            copyto!(fd_v, fd_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
            copyto!(sd_v, sd_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
            copyto!(td_v, td_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
        end
        @inbounds for ip in 1:npar
            copyto!(fd_v, fd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
            copyto!(sd_v, sd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
            copyto!(td_v, td_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
        end
        # function_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(nvar) * "; nx = " * string(nx) * "; ny = " * string(ny) * "; fd_l = " * string(fd_l) * "; fd_c = " * string(fd_c) * "; sd_l = " * string(sd_l) * "; sd_c = " * string(sd_c) * "; td_l = " * string(td_l) * "; td_c = " * string(td_c) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(n, 4*(nx+ny)^2)), d3 = Array{Float64}(zeros(n, 8*(nx+ny)^3))); d1 = " * string(fd_v)[4:end] * "; for ii in 1:length(d1); deriv.d1[fd_l[ii],fd_c[ii]] = d1[ii]; end; d2 =" * string(sd_v)[4:end] * "; for ii in 1:length(d2); deriv.d2[sd_l[ii],sd_c[ii]] = d2[ii]; end; d3 =" * string(td_v)[4:end] * "; @inbounds for ii in 1:length(d3); deriv.d3[td_l[ii],td_c[ii]] = d3[ii]; end; return deriv; end;")
        deriv_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(nvar) * "; nx = " * string(nx) * "; ny = " * string(ny) * "; fd_ci = " * string(fd_ci) * "; sd_ci = " * string(sd_ci) * "; td_ci = " * string(td_ci) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(n, 2*(nx+ny), 2*(nx+ny))), d3 = Array{Float64}(zeros(n, 2*(nx+ny), 2*(nx+ny), 2*(nx+ny)))); d1 = " * string(fd_v)[4:end] * "; @inbounds deriv.d1[fd_ci] = d1; d2 =" * string(sd_v)[4:end] * "; @inbounds deriv.d2[sd_ci] = d2; d3 =" * string(td_v)[4:end] * "; @inbounds deriv.d3[td_ci] = d3; return deriv; end;")    
    end
    eval(deriv_string)
end

function solve_model(model::NamedTuple, deriv::NamedTuple, eta)
    @unpack parameters, estimate, estimation, npar, ns = model
    @unpack x, y, xp, yp, variables, nx, ny, nvar = model 
    @unpack e, ne = model
    @unpack f, nf = model
    @unpack SS, PAR_SS = model
    @unpack flag_order = model

    # eta = diag(sqrt(VAR))

    fd_ = deriv.d1
    sd = deriv.d2

    fx  = @views fd_[:,1:nx]
    fy  = @views fd_[:,nx+1:nvar]
    fxp = @views fd_[:,nvar+1:nvar+nx]
    fyp = @views fd_[:,nvar+nx+1:2*nvar]

    # Complex Schur Decomposition
    F = schur([-fxp -fyp],[fx  fy])

    # # Pick non-explosive (stable) eigenvalues
    slt = abs.(diag(F.T,0)).<abs.(diag(F.S,0))
    nk=sum(slt)

    # # Reorder the system with stable eigs in upper-left
    F = ordschur!(F,slt)
    F.Z
    F.S
    F.T

    # Split up the results appropriately
    z21 = F.Z[nk+1:end,1:nk]
    z11 = F.Z[1:nk,1:nk]

    s11 = F.S[1:nk,1:nk]
    t11 = F.T[1:nk,1:nk]

    indic = 1 # indicates that equilibrium exists and is unique

    # Identify cases with no/multiple solutions
    if nk > size(fx,2)
        # println("The Equilibrium is Locally Indeterminate")
        indic = 2
    elseif nk < size(fx,2)
        # println("No Local Equilibrium Exists")
        indic = 0
    end

    if rank(z11) < nk
        # println("Invertibility cosndition violated")
        indic = 3
    end
    
    if indic == 1
        # Compute the Solution
        z11i = z11\Matrix(I, nk, nk)

        gx = real(z21*z11i)
        hx = real(z11*(s11\t11)*z11i)

        if flag_order > 1
            nfx = @views fd_[:,(1):(nx)]
            nfy = @views fd_[:,(nx+1):(nx+ny)]
            nfxp = @views fd_[:,(nx+ny+1):(nx+ny+nx)]
            nfyp = @views fd_[:,(nx+ny+nx+1):(nx+ny+nx+ny)]

            fypyp = @views sd[:,(nx+ny+nx+1):(nx+ny+nx+ny),(nx+ny+nx+1):(nx+ny+nx+ny)]
            fypxp = @views sd[:,(nx+ny+nx+1):(nx+ny+nx+ny),(nx+ny+1):(nx+ny+nx)]
            fypy = @views sd[:,(nx+ny+nx+1):(nx+ny+nx+ny),(nx+1):(nx+ny)]
            fypx = @views sd[:,(nx+ny+nx+1):(nx+ny+nx+ny),(1):(nx)]
            
            fxpyp = @views sd[:,(nx+ny+1):(nx+ny+nx),(nx+ny+nx+1):(nx+ny+nx+ny)]
            fxpxp = @views sd[:,(nx+ny+1):(nx+ny+nx),(nx+ny+1):(nx+ny+nx)]
            fxpy = @views sd[:,(nx+ny+1):(nx+ny+nx),(nx+1):(nx+ny)]
            fxpx = @views sd[:,(nx+ny+1):(nx+ny+nx),(1):(nx)]

            fyyp = @views sd[:,(nx+1):(nx+ny),(nx+ny+nx+1):(nx+ny+nx+ny)]
            fyxp = @views sd[:,(nx+1):(nx+ny),(nx+ny+1):(nx+ny+nx)]
            fyy = @views sd[:,(nx+1):(nx+ny),(nx+1):(nx+ny)]
            fyx = @views sd[:,(nx+1):(nx+ny),(1):(nx)]

            fxyp = @views sd[:,(1):(nx),(nx+ny+nx+1):(nx+ny+nx+ny)]
            fxxp = @views sd[:,(1):(nx),(nx+ny+1):(nx+ny+nx)]
            fxy = @views sd[:,(1):(nx),(nx+1):(nx+ny)]
            fxx = @views sd[:,(1):(nx),(1):(nx)]


            # fypyp  =   @views reshape(sd[:,ind_yp_yp], nf, ny, ny)
            # fypy   =   @views reshape(sd[:,ind_yp_y], nf, ny, ny)
            # fypxp  =   @views reshape(sd[:,ind_yp_xp], nf, ny, nx)
            # fypx   =   @views reshape(sd[:,ind_yp_x], nf, ny, nx)

            # fyyp   =   @views reshape(sd[:,ind_y_yp], nf, ny, ny)
            # fyy    =   @views reshape(sd[:,ind_y_y], nf, ny, ny)
            # fyxp   =   @views reshape(sd[:,ind_y_xp], nf, ny, nx)
            # fyx    =   @views reshape(sd[:,ind_y_x], nf, ny, nx)

            # fxpyp  =   @views reshape(sd[:,ind_xp_yp], nf, nx, ny)
            # fxpy   =   @views reshape(sd[:,ind_xp_y], nf, nx, ny)
            # # nfxpxp  =   @views reshape(sd[:,ind_xp_xp]
            # fxpxp  =   @views reshape(sd[:,ind_xp_xp], nf, nx, nx)
            # fxpx   =   @views reshape(sd[:,ind_xp_x], nf, nx, nx)

            # fxyp   =   @views reshape(sd[:,ind_x_yp], nf, nx, ny)
            # fxy    =   @views reshape(sd[:,ind_x_y], nf, nx, ny)
            # fxxp   =   @views reshape(sd[:,ind_x_xp], nf, nx, nx)
            # fxx    =   @views reshape(sd[:,ind_x_x], nf, nx, nx)

            nfypyp  =   @views reshape(permutedims(fypyp, [2; 1; 3]), nf*ny, ny)
            nfypy   =   @views reshape(permutedims(fypy, [2; 1; 3]), nf*ny, ny)
            nfypxp  =   @views reshape(permutedims(fypxp, [2; 1; 3]), nf*ny, nx)
            nfypx   =   @views reshape(permutedims(fypx, [2; 1; 3]), nf*ny, nx)

            nfyyp   =   @views reshape(permutedims(fyyp, [2; 1; 3]), nf*ny, ny)
            nfyy    =   @views reshape(permutedims(fyy, [2; 1; 3]), nf*ny, ny)
            nfyxp   =   @views reshape(permutedims(fyxp, [2; 1; 3]), nf*ny, nx)
            nfyx    =   @views reshape(permutedims(fyx, [2; 1; 3]), nf*ny, nx)

            nfxpyp  =   @views reshape(permutedims(fxpyp, [2; 1; 3]), nf*nx, ny)
            nfxpy   =   @views reshape(permutedims(fxpy, [2; 1; 3]), nf*nx, ny)
            nfxpxp  =   @views reshape(permutedims(fxpxp, [2; 1; 3]), nf*nx, nx)
            nfxpx   =   @views reshape(permutedims(fxpx, [2; 1; 3]), nf*nx, nx)

            nfxyp   =   @views reshape(permutedims(fxyp, [2; 1; 3]), nf*nx, ny)
            nfxy    =   @views reshape(permutedims(fxy, [2; 1; 3]), nf*nx, ny)
            nfxxp   =   @views reshape(permutedims(fxxp, [2; 1; 3]), nf*nx, nx)
            nfxx    =   @views reshape(permutedims(fxx, [2; 1; 3]), nf*nx, nx)

            # nx=size(nfx,2)
            # ny = size(nfy,2)
            # n = nx + ny
            
            A = kron(Matrix{Float64}(I, nx+ny, nx+ny),hx'*gx')*(nfypyp*gx*hx + nfypxp*hx + nfypy*gx + nfypx) + 
                kron(Matrix{Float64}(I, nx+ny, nx+ny),gx')    *(nfyyp *gx*hx + nfyxp *hx + nfyy *gx + nfyx ) +     
                kron(Matrix{Float64}(I, nx+ny, nx+ny),hx')    *(nfxpyp*gx*hx + nfxpxp*hx + nfxpy*gx + nfxpx) + 
                                    (nfxyp *gx*hx + nfxxp *hx + nfxy *gx + nfxx);   
            B = kron(nfyp, hx') ; 
            C = kron(nfy, Matrix{Float64}(I, nx, nx))
            D = kron(nfyp*gx, Matrix{Float64}(I, nx, nx))+ kron(nfxp, Matrix{Float64}(I, nx, nx))

            Qq = -[ kron(hx',B)+kron(Matrix{Float64}(I, nx, nx),C) kron(Matrix{Float64}(I, nx, nx), D)] \ (A[:])
            # inv(-[ kron(hx',B)+kron(Matrix{Float64}(I, nx, nx),C) kron(Matrix{Float64}(I, nx, nx), D)]) * (A[:])

            gxx = permutedims(reshape(Qq[1:nx^2*ny],nx,ny,nx),[2 1 3]) #this reshaping is done so that we get the output in the same order as we used to
            hxx = permutedims(reshape(Qq[nx^2*ny+1:end],nx,nx,nx),[2 1 3]) #reshape to get the same as in the previous version version of gxx_hxx.m

            Qh = zeros(nf,nx)
            Qg = zeros(nf,ny)
            q = zeros(nf)

            if typeof(eta) != Vector{Float64}
                for ii in 1:nf

                    # First Term
                    Qh[ii,:] = fyp[ii,:]' * gx;
                    
                    # Second Term
                    q[ii] = sum(diag((permutedims(fypyp[ii,:,:],[2 1])*gx*eta)'*gx*eta));
                    # println(q[ii])

                    # Third Term
                    q[ii] = q[ii] + sum(diag((fypxp[ii,:,:]*eta)'*gx*eta));
                    # println(q[ii])

                    # Fourth Term
                    q[ii] =  q[ii] + sum(diag(( reshape(fyp[ii,:]'*reshape(gxx,ny,nx^2),nx,nx)*eta)'*eta));
                    # println(q[ii])

                    # Fifth Term
                    Qg[ii,:] = fyp[ii,:];
                    
                    # Sixth Term
                    Qg[ii,:] = Qg[ii,:] + fy[ii,:];
                    
                    # Seventh Term
                    Qh[ii,:] = Qh[ii,:] + fxp[ii,:];
                    
                    # Eighth Term
                    q[ii] = q[ii] + sum(diag((fxpyp[ii,:,:]*gx*eta)'*eta));
                    
                    # Nineth Term
                    q[ii] = q[ii] + sum(diag((permutedims(fxpxp[ii,:,:],[2 1])*eta)'*eta));
                end
            else
                for ii in 1:nf

                    # First Term
                    Qh[ii,:] = fyp[ii,:]' * gx;
                    
                    # Second Term
                    q[ii] = sum(diag([(permutedims(fypyp[ii,:,:],[2 1])*gx*eta)'*gx*eta][:,:]));
                    # println(q[ii])

                    # Third Term
                    q[ii] = q[ii] + sum(diag([(fypxp[ii,:,:]*eta)'*gx*eta][:,:]));
                    # println(q[ii])

                    # Fourth Term
                    q[ii] =  q[ii] + sum(diag([( reshape(fyp[ii,:]'*reshape(gxx,ny,nx^2),nx,nx)*eta)'*eta][:,:]));
                    # println(q[ii])

                    # Fifth Term
                    Qg[ii,:] = fyp[ii,:];
                    
                    # Sixth Term
                    Qg[ii,:] = Qg[ii,:] + fy[ii,:];
                    
                    # Seventh Term
                    Qh[ii,:] = Qh[ii,:] + fxp[ii,:];
                    
                    # Eighth Term
                    q[ii] = q[ii] + sum(diag([(fxpyp[ii,:,:]*gx*eta)'*eta][:,:]));
                    
                    # Nineth Term
                    q[ii] = q[ii] + sum(diag([(permutedims(fxpxp[ii,:,:],[2 1])*eta)'*eta][:,:]));
                end
            end    
            x = -([Qg Qh])\q;
                
            gss = x[1:ny]
            hss = x[ny+1:end]
            return (gx = gx, hx = hx, gxx = gxx, hxx = hxx, gss = gss, hss = hss,
                    qzflag = indic)
        else
            return (gx = gx, hx = hx,
                    qzflag = indic)
        end
    else
        return (gx = [], hx = [], qzflag = indic)
    end
end

function simulate_model(model::NamedTuple, sol_mat::NamedTuple, nsim::Int)
    if model.flag_order > 1
        sim_shocks = randn(nsim+10000, model.ne)
        sim_x_f = zeros(nsim+10000, model.nx)
        sim_x_s = zeros(nsim+10000, model.nx)
        sim_y = zeros(nsim+10000, model.ny)
        for t in 1:nsim+10000
            for i in 1:model.ny
                sim_y[t,i] = sol_mat.gx[i,:]' * (sim_x_f[t,:]+sim_x_s[t,:]) + 1/2 * sim_x_f[t,:]' * sol_mat.gxx[i,:,:] * sim_x_f[t,:] + 1/2 * sol_mat.gss[i] 
            end
            if t < nsim+10000
                for i=1:nx
                    sim_x_f[t+1,i] = sol_mat.hx[i,:]' * sim_x_f[t,:] + eta[i,:]'*sim_shocks[t,:]
                    sim_x_s[t+1,i] = sol_mat.hx[i,:]' * sim_x_s[t,:] + 1/2 * sim_x_f[t,:]' * sol_mat.hxx[i,:,:] * sim_x_f[t,:] + 1/2 * sol_mat.hss[i]
                end
            end
        end
        return [sim_x_f + sim_x_s sim_y]
    else
        sim_shocks = randn(nsim+10000, model.ne)
        sim_x = zeros(nsim+10000, model.nx)
        sim_y = zeros(nsim+10000, model.ny)
        for t in 1:nsim+10000
            for i in 1:model.ny
                sim_y[t,i] = sol_mat.gx[i,:]' * sim_x[t,:]
            end
            if t < nsim+10000
                for i=1:nx
                    sim_x[t+1,i] = sol_mat.hx[i,:]' * sim_x[t,:] + eta[i,:]'*sim_shocks[t,:]
                end
            end
        end
        return [sim_x sim_y]
    end
end

function impulseresponse_model(model::NamedTuple, sol_mat::NamedTuple, nsim::Int)
    sim_shocks = zeros(nsim+10000, model.ne)
    sim_shocks[1,1] = 1
    if model.flag_order > 1
        sim_x_f = zeros(nsim+10000, model.nx)
        sim_x_s = zeros(nsim+10000, model.nx)
        sim_y = zeros(nsim+10000, model.ny)
        for t in 1:nsim+10000
            for i in 1:model.ny
                sim_y[t,i] = sol_mat.gx[i,:]' * (sim_x_f[t,:]+sim_x_s[t,:]) + 1/2 * sim_x_f[t,:]' * sol_mat.gxx[i,:,:] * sim_x_f[t,:] + 1/2 * sol_mat.gss[i] 
            end
            if t < nsim+10000
                for i=1:nx
                    sim_x_f[t+1,i] = sol_mat.hx[i,:]' * sim_x_f[t,:] + eta[i,:]'*sim_shocks[t,:]
                    sim_x_s[t+1,i] = sol_mat.hx[i,:]' * sim_x_s[t,:] + 1/2 * sim_x_f[t,:]' * sol_mat.hxx[i,:,:] * sim_x_f[t,:] + 1/2 * sol_mat.hss[i]
                end
            end
        end
        return [sim_x_f + sim_x_s sim_y]
    else
        sim_x = zeros(nsim+10000, model.nx)
        sim_y = zeros(nsim+10000, model.ny)
        for t in 1:nsim+10000
            for i in 1:model.ny
                sim_y[t,i] = sol_mat.gx[i,:]' * sim_x[t,:]
            end
            if t < nsim+10000
                for i=1:nx
                    sim_x[t+1,i] = sol_mat.hx[i,:]' * sim_x[t,:] + eta[i,:]'*sim_shocks[t,:]
                end
            end
        end
        return [sim_x sim_y]
    end
end

function smc_rwmh(model::NamedTuple, data::Matrix, PAR::Vector, c::Float64, npart::Int, nphi::Int, lam::Int)
    # initial_para = [0.9; 0.15]
    # npara=size(initial_para,1)

    phi_bend = true
    nobs = size(data,2)
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
    zhat    = zeros(nphi)             # normalization constant
    nresamp = 0                         # record # of iteration resampled

    csim    = zeros(nphi) # scale parameter
    ESSsim  = zeros(nphi) # ESS
    acptsim = zeros(nphi) # average acceptance rate
    rsmpsim = zeros(nphi) # 1 if re-sampled

    # SMC algorithm starts here

    # ------------------------------------------------------------------------
    # Initialization: Draws from the prior
    # ------------------------------------------------------------------------
    println("SMC starts ... ")
    # drawing initial values from the prior distributions
    # parasim = zeros(nphi, npart, model.ns) # parameter draws
    parasim = initial_draw(model, npart, nphi)
    # println(parasim[1,1,:])

    wtsim[:, 1]    = 1.0./npart .* ones(npart,1)      # initial weight is equal weights
    zhat[1]        = sum(wtsim[:,1])

    # Posterior values at prior draws
    loglh  = zeros(npart) #log-likelihood
    logpost = zeros(npart) #log-posterior
    @inbounds for i in 1:npart
        p0 = parasim[1,i,:]
        loglh[i] = objfun(data, p0, phi_smc, model, nobs, PAR)
        prior_val = logpriors(model, p0)
        logpost[i]=loglh[i]+prior_val            
    end
    # println(loglh)
    loglh[isnan.(loglh)] .= -1e50
    loglh[isinf.(loglh)] .= -1e50
    logpost[isnan.(logpost)] .= -1e50
    logpost[isinf.(logpost)] .= -1e50
            
    # ------------------------------------------------------------------------
    # Recursion: For n=2,...,N_[\phi]
    # ------------------------------------------------------------------------
    estimMean = zeros(nphi, model.ns)

    println("SMC recursion starts ... ")

    @inbounds for i in 2:nphi
            
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
        ESS = 1.0/sum(wtsim[:, i].^2.0) # Effective sample size()

        if (ESS .< npart/2)
            id=Vector{Int64}(undef, npart)
            @inbounds Threads.@threads for ii=1:npart   
                # id[ii]=sample(1:npart, ProbabilityWeights(wtsim[:,i]), true)[[1]][1]
                # id[ii]=sample(1:npart, AnalyticWeights(wtsim[:,i]))
                id[ii]=sample(Random.GLOBAL_RNG, 1:npart, weights(wtsim[:,i]), true)[[1]][1]
            end
            parasim[i-1, :, :] = parasim[i-1, id, :]
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
        
        wght =  wtsim[:, i].*ones(npart,model.ns)
        
        mu      = sum(para.*wght) # mean()
        z       = (para - mu.*ones(npart, model.ns))
        R       = (z.*wght)'*z       # covariance
        # if ESS .< npart/2  || det(R) <= 0.0 
        # eps = 1.e-6
        # pdef = sum(real(eigen(R).values) .< eps)
        # while pdef > 0.0
        #     leftR = real(eigen(R).vectors)
        #     tauR = real(eigen(R).values)
        #     tauR[tauR .< eps] .= eps
        #     R = leftR * diagm(tauR) *leftR'
        #     pdef = sum(tauR .< eps)
        # end
        eps = 1.e-3
        for t in 1:10
            leftR = real(eigen(R).vectors)
            tauR = real(eigen(R).values)
            tauR[tauR .< eps] .= eps
            R = leftR * diagm(tauR) *leftR'
        end
        # end        
        Rdiag   = Diagonal(R)#diag(diag(R)) # covariance with diag elements
        Rchol   = cholesky(Hermitian(R)).U
        Rchol2  = sqrt(Rdiag)
        
        estimMean[i,:] .= mu
        
        # Particle mutation [Algorithm 2]
        temp_acpt = zeros(npart,1) #initialize accpetance indicator
        
        # @inbounds Threads.@threads for j = 1:npart #iteration over particles
        # j = 1
        for j = 1:npart #iteration over particles
            ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(model, data, para[j,:]', loglh[j], logpost[j], c, R, model.ns, phi_smc, nobs, i, PAR)
                        
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
 
    println("mu: $(mu)") 
    println("sig: $(sig)")
end

function smc_rwmh(model::NamedTuple, data::Matrix{Float64}, PAR::Vector{Float64}, PAR_SS::Vector{Float64}, VAR::Array{Float64}, SS::Vector{Float64}, deriv::NamedTuple, sol_mat::NamedTuple, c::Float64, npart::Int, nphi::Int, lam::Int)
    # initial_para = [0.9; 0.15]
    # npara=size(initial_para,1)

    phi_bend = true
    ny, nobs = size(data)
    acpt=0.25
    trgt=0.25
    
    DD = zeros(nobs)                                   # DD = 0.0*ones(nobs,1); mean_obs, constants in the observable equation
    ZZ = Matrix{Float64}(I, nobs, model.ny)     # ZZ = B*gx;
    HH = 0.0001*Matrix{Float64}(I, nobs, nobs)

    At = Array{Float64}(zeros(model.nx))
    Pt = Array{Float64}(I, model.nx, model.nx)
    loglhvec = Array{Float64}(zeros(ny))
    TTPtTT= similar(Pt)
    KFK= similar(Pt)

    aux = nobs*0.5*ny*log(2.0*π)
    
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
    zhat    = zeros(nphi)             # normalization constant
    nresamp = 0                         # record # of iteration resampled

    csim    = zeros(nphi) # scale parameter
    ESSsim  = zeros(nphi) # ESS
    acptsim = zeros(nphi) # average acceptance rate
    rsmpsim = zeros(nphi) # 1 if re-sampled

    p0 = zeros(model.ns)
    # SMC algorithm starts here

    # ------------------------------------------------------------------------
    # Initialization: Draws from the prior
    # ------------------------------------------------------------------------
    println("SMC starts ... ")
    # drawing initial values from the prior distributions
    # parasim = zeros(nphi, npart, model.ns) # parameter draws
    parasim = initial_draw(model, npart, nphi)

    wtsim[:, 1]    .= 1.0./npart .* ones(npart)      # initial weight is equal weights
    zhat[1]        = sum(wtsim[:,1])

    # Posterior values at prior draws
    loglh  = zeros(npart) #log-likelihood
    logpost = zeros(npart) #log-posterior
    prior_val = 0.0

    @time @inbounds for i in 1:npart
        p0 .= parasim[1,i,:]
        loglh[i] = objfun(data, p0, model, PAR, PAR_SS, VAR, SS, deriv, sol_mat, DD, ZZ, HH, At, Pt, loglhvec, TTPtTT, KFK, aux)
        prior_val = logpriors(model, p0)
        logpost[i] = loglh[i] + prior_val            
    end

    loglh[isnan.(loglh)] .= -1e50
    loglh[isinf.(loglh)] .= -1e50
    logpost[isnan.(logpost)] .= -1e50
    logpost[isinf.(logpost)] .= -1e50
            
    # ------------------------------------------------------------------------
    # Recursion: For n=2,...,N_[\phi]
    # ------------------------------------------------------------------------
    estimMean = zeros(nphi, model.ns)

    para = zeros(npart, model.ns)
    wght = zeros(npart,model.ns)
    mu = 0.0
    z = zeros(npart,model.ns)
    R = zeros(model.ns,model.ns)

    ind_par = zeros(model.ns)
    ind_loglh = 0.0
    ind_post = 0.0 
    ind_acpt = 0

    println("SMC recursion starts ... ")

    @inbounds for i in 2:nphi
            
        #allReal = isreal(loglh)
        loglh = real(loglh)
        #-----------------------------------
        # (a) Correction
        #-----------------------------------
        # incremental weights
        incwt = exp.((phi_smc[i]-phi_smc[i-1])*loglh)
            
        # update weights
        wtsim[:, i] .= (@view wtsim[:, i-1]).*incwt
        zhat[i]     = sum(wtsim[:, i])
        
        # normalize weights
        wtsim[:, i] .= (@view wtsim[:, i])/zhat[i]

        #-----------------------------------
        # (b) Selection
        #-----------------------------------
        ESS = 1.0./sum(wtsim[:, i].^2.0) # Effective sample size()

        if (ESS .< npart/2)
            id=Vector{Int64}(undef, npart)
            @inbounds Threads.@threads for ii=1:npart   
                # id[ii]=sample(1:npart, ProbabilityWeights(wtsim[:,i]), true)[[1]][1]
                id[ii]=sample(1:npart, AnalyticWeights(wtsim[:,i]))
                # id[ii]=sample(Random.GLOBAL_RNG, 1:npart, weights(wtsim[:,i]), true)[[1]][1]
            end
            parasim[i-1, :, :] = @view parasim[i-1, id, :]
            loglh            = loglh[id]
            logpost          = logpost[id]
            wtsim[:, i]      = 1.0/npart .* ones(npart)  # resampled weights are equal weights
            nresamp          = nresamp + 1
            rsmpsim[i]       = 1    
        end
        
        #--------------------------------------------------------
        # (c) Mutuation
        #--------------------------------------------------------
        # Adapting the transition kernel
        c = c*(0.95 + 0.10*exp(16.0*(acpt-trgt))/(1 + exp(16.0*(acpt-trgt))))
        
        # Calculate estimates of mean & variance
        para = @view parasim[i-1, :, :]
        
        wght .=  (@view wtsim[:, i]).*ones(npart,model.ns)
        
        mu      = sum(para.*wght) # mean()
        z       = (para - mu.*ones(npart, model.ns))
        R       = (z.*wght)'*z       # covariance
        # eps = 1.e-3
        # for t in 1:10
        #     leftR = real(eigen(R).vectors)
        #     tauR = real(eigen(R).values)
        #     tauR[tauR .< eps] .= eps
        #     R = leftR * diagm(tauR) *leftR'
        # end
        # Rdiag   = Diagonal(R)#diag(diag(R)) # covariance with diag elements
        # Rchol   = cholesky(Hermitian(R)).U
        # Rchol2  = sqrt(Rdiag)
        
        estimMean[i,:] .= mu
        
        # Particle mutation [Algorithm 2]
        temp_acpt = zeros(npart) #initialize accpetance indicator
        
        @time @inbounds for j = 1:npart #iteration over particles
            ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(model, data, (@view para[j,:]), loglh[j], logpost[j], c, R, model.ns, 
                                                                    PAR, PAR_SS, VAR, SS, deriv, sol_mat, 
                                                                    DD, ZZ, HH, At, Pt, loglhvec, TTPtTT, KFK, aux)
            parasim[i,j,:] .= ind_para
            loglh[j]       = ind_loglh
            logpost[j]     = ind_post
            temp_acpt[j] = ind_acpt
            
        end
        acpt = mean(temp_acpt) # update average acceptance rate
        
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
 
    println("mu: $(mu)") 
    println("sig: $(sig)")
end

function smc_rwmh_threads(model::NamedTuple, data::Matrix{Float64}, PAR::Vector{Float64}, PAR_SS::Vector{Float64}, VAR::Array{Float64}, SS::Vector{Float64}, deriv::NamedTuple, sol_mat::NamedTuple, c::Float64, npart::Int, nphi::Int, lam::Int)
    # initial_para = [0.9; 0.15]
    # npara=size(initial_para,1)

    phi_bend = true
    ny, nobs = size(data)
    acpt=0.25
    trgt=0.25
    
    DD = zeros(nobs)                                   # DD = 0.0*ones(nobs,1); mean_obs, constants in the observable equation
    ZZ = Matrix{Float64}(I, nobs, model.ny)     # ZZ = B*gx;
    HH = 0.0001*Matrix{Float64}(I, nobs, nobs)

    At = Array{Float64}(zeros(model.nx))
    Pt = Array{Float64}(I, model.nx, model.nx)
    loglhvec = Array{Float64}(zeros(ny))
    TTPtTT= similar(Pt)
    KFK= similar(Pt)

    aux = nobs*0.5*ny*log(2.0*π)
    
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
    zhat    = zeros(nphi)             # normalization constant
    nresamp = 0                         # record # of iteration resampled

    csim    = zeros(nphi) # scale parameter
    ESSsim  = zeros(nphi) # ESS
    acptsim = zeros(nphi) # average acceptance rate
    rsmpsim = zeros(nphi) # 1 if re-sampled

    p0 = zeros(model.ns)
    # SMC algorithm starts here

    # ------------------------------------------------------------------------
    # Initialization: Draws from the prior
    # ------------------------------------------------------------------------
    println("SMC starts ... ")
    # drawing initial values from the prior distributions
    # parasim = zeros(nphi, npart, model.ns) # parameter draws
    parasim = initial_draw(model, npart, nphi)

    wtsim[:, 1]    .= 1.0./npart .* ones(npart)      # initial weight is equal weights
    zhat[1]        = sum(wtsim[:,1])

    # Posterior values at prior draws
    loglh  = zeros(npart) #log-likelihood
    logpost = zeros(npart) #log-posterior
    prior_val = 0.0

    @inbounds Threads.@threads for i in 1:npart
        p0 .= parasim[1,i,:]
        loglh[i] = objfun(data, p0, model, PAR, PAR_SS, VAR, SS, deriv, sol_mat, DD, ZZ, HH, At, Pt, loglhvec, TTPtTT, KFK, aux)
        prior_val = logpriors(model, p0)
        logpost[i] = loglh[i] + prior_val            
    end

    loglh[isnan.(loglh)] .= -1e50
    loglh[isinf.(loglh)] .= -1e50
    logpost[isnan.(logpost)] .= -1e50
    logpost[isinf.(logpost)] .= -1e50
            
    # ------------------------------------------------------------------------
    # Recursion: For n=2,...,N_[\phi]
    # ------------------------------------------------------------------------
    estimMean = zeros(nphi, model.ns)

    para = zeros(npart, model.ns)
    wght = zeros(npart,model.ns)
    mu = 0.0
    z = zeros(npart,model.ns)
    R = zeros(model.ns,model.ns)

    ind_par = zeros(model.ns)
    ind_loglh = 0.0
    ind_post = 0.0 
    ind_acpt = 0

    println("SMC recursion starts ... ")

    @inbounds for i in 2:nphi
            
        #allReal = isreal(loglh)
        loglh = real(loglh)
        #-----------------------------------
        # (a) Correction
        #-----------------------------------
        # incremental weights
        incwt = exp.((phi_smc[i]-phi_smc[i-1])*loglh)
            
        # update weights
        wtsim[:, i] .= (@view wtsim[:, i-1]).*incwt
        zhat[i]     = sum(wtsim[:, i])
        
        # normalize weights
        wtsim[:, i] .= (@view wtsim[:, i])/zhat[i]

        #-----------------------------------
        # (b) Selection
        #-----------------------------------
        ESS = 1.0./sum(wtsim[:, i].^2.0) # Effective sample size()

        if (ESS .< npart/2)
            id=Vector{Int64}(undef, npart)
            @inbounds Threads.@threads for ii=1:npart   
                # id[ii]=sample(1:npart, ProbabilityWeights(wtsim[:,i]), true)[[1]][1]
                id[ii]=sample(1:npart, AnalyticWeights(wtsim[:,i]))
                # id[ii]=sample(Random.GLOBAL_RNG, 1:npart, weights(wtsim[:,i]), true)[[1]][1]
            end
            parasim[i-1, :, :] = @view parasim[i-1, id, :]
            loglh            = loglh[id]
            logpost          = logpost[id]
            wtsim[:, i]      = 1.0/npart .* ones(npart)  # resampled weights are equal weights
            nresamp          = nresamp + 1
            rsmpsim[i]       = 1    
        end
        
        #--------------------------------------------------------
        # (c) Mutuation
        #--------------------------------------------------------
        # Adapting the transition kernel
        c = c*(0.95 + 0.10*exp(16.0*(acpt-trgt))/(1 + exp(16.0*(acpt-trgt))))
        
        # Calculate estimates of mean & variance
        para = @view parasim[i-1, :, :]
        
        wght .=  (@view wtsim[:, i]).*ones(npart,model.ns)
        
        mu      = sum(para.*wght) # mean()
        z       = (para - mu.*ones(npart, model.ns))
        R       = (z.*wght)'*z       # covariance
        # eps = 1.e-3
        # for t in 1:10
        #     leftR = real(eigen(R).vectors)
        #     tauR = real(eigen(R).values)
        #     tauR[tauR .< eps] .= eps
        #     R = leftR * diagm(tauR) *leftR'
        # end
        # Rdiag   = Diagonal(R)#diag(diag(R)) # covariance with diag elements
        # Rchol   = cholesky(Hermitian(R)).U
        # Rchol2  = sqrt(Rdiag)
        
        estimMean[i,:] .= mu
        
        # Particle mutation [Algorithm 2]
        temp_acpt = zeros(npart) #initialize accpetance indicator
        
        @inbounds Threads.@threads for j = 1:npart #iteration over particles
            ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(model, data, (@view para[j,:]), loglh[j], logpost[j], c, R, model.ns, 
                                                                    PAR, PAR_SS, VAR, SS, deriv, sol_mat, 
                                                                    DD, ZZ, HH, At, Pt, loglhvec, TTPtTT, KFK, aux)
            parasim[i,j,:] .= ind_para
            loglh[j]       = ind_loglh
            logpost[j]     = ind_post
            temp_acpt[j] = ind_acpt
            
        end
        acpt = mean(temp_acpt) # update average acceptance rate
        
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
 
    println("mu: $(mu)") 
    println("sig: $(sig)")
end

function initial_draw(model::NamedTuple, npart::Int, nphi::Int)

        parasim = zeros(nphi, npart, model.ns) # parameter draws
        
        for i in 1:model.ns
            parasim[1, :, i] = rand(model.priors[i].d, npart)
            parasim[1, 1, i] = model.priors[i].iv
        end
        
    # # Parameter1: Truncated normal
    #     mean1 = 0.75
    #     std1 = 0.25
    #     d1 = Normal(mean1, std1) #Normal{Float64}(μ=0.16, σ=0.05)
    #     td1 = truncated(d1, 0.0, 1.0)
    #     parasim[1, :, 1] = rand(td1, npart)
    
    # # Parameter2: Inverse gamma
    #     shape2 = 5.0
    #     scale2 = 0.25
    #     d2 = InverseGamma(shape2, scale2) # Normal{Float64}(μ=0.16, σ=0.05)
    #     parasim[1, :, 2] = rand(d2, npart)
    
    # # [...]
    
    return parasim
end

function objfun(yy::Matrix{Float64}, p0, model::NamedTuple, PAR::Vector{Float64}, PAR_SS::Vector{Float64}, VAR::Array{Float64}, SS::Vector{Float64}, deriv::NamedTuple, sol_mat::NamedTuple,
                DD::Array{Float64}, ZZ::Matrix{Float64}, HH::Matrix{Float64}, 
                At::Vector{Float64}, Pt::Matrix{Float64}, loglhvec::Vector{Float64}, TTPtTT::Matrix{Float64}, KFK::Matrix{Float64}, aux::Float64)
    
    @inbounds for ii in 1:model.ns
        PAR[model.estimation[ii]] = p0[ii]
    end
    VAR     =   eval_ShockVAR(PAR)
    PAR_SS  =   eval_PAR_SS(PAR)
    # println("Ok")
    if !model.flag_SSsolver
        SS      =   eval_SS(PAR_SS)
    # else
        # SS      =   SS_solver(model, PAR)
        # SS = zeros(model.nx+model.ny+model.nx+model.ny)
    end
    deriv   =   eval_deriv(PAR_SS, SS)

    sol_mat     =   solve_model(model, deriv, VAR)
    # print(sol_mat)

    if sol_mat.qzflag .== 1
        # ny, nobs = size(data)
        # TT = sol_mat.hx                                  # TT = hx;
        # # println(TT)
        # # sleep(10)
        # RQR = VAR*VAR'                                                # RQR = VAR;
        # # println(RQR)
        # # sleep(10)
        # DD = zeros(nobs)                                   # DD = 0.0*ones(nobs,1); mean_obs, constants in the observable equation
        # # println(nobs)
        # # println(DD)
        # # sleep(10)
        # ZZ = Matrix{Float64}(I, nobs, model.ny)*sol_mat.gx     # ZZ = B*gx;
        # # println(ZZ)
        # # sleep(10)
        # HH =   0.0001*Matrix{Float64}(I, nobs, nobs)  # HH = [0.0001 0.0 0.0 ;0.0 0.0001 0.0 ; 0.0 0.0 0.0001]#zeros(3,3)# diag(nVAR_me);   m.e. must be in STD
        # # println(HH)
        # # sleep(10)
        
        # At = Array{Float64}(zeros(model.nx))
        # Pt = Array{Float64}(I, model.nx, model.nx)
        # loglhvec = Array{Float64}(zeros(ny))
        # TTPtTT= Array{Float64}(I, model.nx, model.nx)
        # KFK= Array{Float64}(I, model.nx, model.nx)

        # return @time kf(yy, TT, RQR, DD, ZZ, HH)
        aaa = kf(yy,
                    sol_mat.hx,                                  # TT = hx;
                    VAR*VAR',                                                # RQR = VAR;
                    DD,
                    ZZ*sol_mat.gx,
                    HH,
                    At, Pt, loglhvec, TTPtTT, KFK, aux)   # HH = [0.0001 0.0 0.0 ;0.0 0.0001 0.0 ; 0.0 0.0 0.0001]#zeros(3,3)# diag(nVAR_me);   m.e. must be in STD
        # sleep(10)
        return aaa
    else
        return -1e16
    end
end


function kf(y::Matrix{Float64}, TT::Matrix{Float64}, RQR::Matrix{Float64}, DD::Array{Float64}, ZZ::Matrix{Float64}, HH::Matrix{Float64}, 
            At::Vector, Pt::Matrix, loglhvec::Vector{Float64}, TTPtTT::Matrix{Float64}, KFK::Matrix{Float64}, aux::Float64)
    # println(DD)
    # println(y)
    # println(TT)
    # println(RQR)
    # println(DD)
    # println(ZZ)
    # println(HH)
    nobs, ny = size(y)
    # ns = size(TT,2)
    
    # At = Array{Float64}(zeros(ns))#zeros(ns, 1)
    
    TT_old = copy(TT)
    RQR_old = copy(RQR)
    # P_10_old=Array{Float64}(I, size(TT))
    # loglhvec=zeros(nobs)
    P_10 = copy(Pt)
    diferenz=0.1

    # @time while diferenz>1e-25
    while diferenz>1e-25
        Pt = TT_old*P_10*TT_old' 
        Pt += RQR_old
        diferenz = maximum(abs.(Pt-P_10))
        RQR_old = TT_old*RQR_old*TT_old' + RQR_old
        TT_old = TT_old * TT_old
        P_10 = Pt
    end    #while diferenz
    # println(P_10)
    # Pt = copy(P_10)
    # loglh = 0.0
    # yaux = similar(DD)
    # ZPZ = similar(HH)
    # TTPt = similar(Pt)
    # TAt = similar(At)
    # KiF = similar(At)
    # PtZZ = similar(Pt*ZZ')
    # Kt = similar(Pt*ZZ')
    TTPtTT = copy(Pt)
    KFK = copy(Pt)
    # iFtnut = similar(DD)
    
    yaux, yhat, TTPt, Kt, iFtnut = similar(DD), similar(DD), similar(Pt), similar(Pt*ZZ'), similar(DD)
    Ft, iFt, nut = zeros(ny,ny), zeros(ny,ny), zeros(ny)
    dFt = 0.0
    # if isnan(sum(At))
    #     println(At)
    # end

    # @time @inbounds for i in 1:nobs
    @inbounds for i in 1:nobs
        mul!(yaux,ZZ,At)
        # println(DD)
        # sleep(10)
        yhat = yaux + DD
        # if isnan(sum(yhat))
        #     println(i)
        # end
        # println(y[i,:])
        # println(yhat)
        # sleep(10)
        nut = y[i, :] - yhat
        # println(nut)
        # sleep(10)
        # println(typeof(ZZ))
        # println(typeof(Pt))
        # println(typeof(PtZZ))
        # @time @views mul!(PtZZ,Pt,ZZ')
        # println(typeof(ZPZ))
        # @time @views mul!(ZPZ,ZZ,PtZZ)
        # @time @views ZPZ = ZZ * Pt * ZZ'
        Ft = ZZ * Pt * ZZ' + HH #ZZ*Pt*ZZ' .+ HH
        # println(Ft)
        # sleep(10)
        Ft = 0.5*(Ft + Ft')
        # println(Ft)
        # sleep(10)
        dFt = det(Ft)
        iFt = inv(Ft)
        # println(Ft)
        # sleep(10)
        mul!(iFtnut,iFt,nut)
        # println(iFtnut)
        # sleep(10)
        
        # if isnan(sum(nut*iFtnut))
        #     println(i)
        # end
        loglhvec[i] = -0.5*(log(dFt) + (iFtnut'*nut))
        # loglhvec[i] = -mean([log(dFt); (0.5*iFtnut'*nut)])
        # println(loglhvec[i])
        # sleep(10)
        mul!(TTPt,TT,Pt)
        mul!(Kt,TTPt,ZZ')
        # @time @views TAt = TT * At
        # @time @views KiF = Kt * iFtnut
        # @time @views At = TAt + KiF
        At = TT * At + Kt * iFtnut
        mul!(TTPtTT,TTPt,TT')
        mul!(KFK,Kt,(iFt*Kt'))
        # KFK = Kt * iFt * Kt'
        Pt = TTPtTT - KFK + RQR
        # sleep(10)
        # At, Pt, loglhvec[i] = kf_inner_loop(yaux,ZZ,At,DD, yhat, nut, Pt, Ft, HH, iFtnut, TTPt, TT, Kt, TTPtTT, KFK, iFt, RQR, y[i,:])
    end
    # loglh = sum(loglhvec) - nobs*0.5*ny*log(2.0*π)
    # return loglh
    # return sum(loglhvec) - nobs*0.5*ny*log(2.0*π)
    # return sum(loglhvec) - aux
    sum(loglhvec) - aux
end    

# function kf(y::Matrix{Float64}, TT::Matrix{Float64}, RQR::Matrix{Float64}, DD::Array{Float64}, ZZ::Matrix{Float64}, HH::Matrix{Float64}, 
#             At::Vector, Pt::Matrix, loglhvec::Vector{Float64}, TTPtTT::Matrix{Float64}, KFK::Matrix{Float64}, aux::Float64)
#     # println(DD)
#     # println(y)
#     # println(TT)
#     # println(RQR)
#     # println(DD)
#     # println(ZZ)
#     # println(HH)

#     @time nobs, ny = size(y)
#     # ns = size(TT,2)
    
#     # At = Array{Float64}(zeros(ns))#zeros(ns, 1)
#     @time begin
#         TT_old = copy(TT)
#         RQR_old = copy(RQR)
#         # P_10_old=Array{Float64}(I, size(TT))
#         # loglhvec=zeros(nobs)
#         P_10 = copy(Pt)
#         diferenz=0.1
#     end
#     @time while diferenz>1e-25
#     # while diferenz>1e-25
#         Pt = TT_old*P_10*TT_old' 
#         Pt += RQR_old
#         diferenz = maximum(abs.(Pt-P_10))
#         RQR_old = TT_old*RQR_old*TT_old' + RQR_old
#         TT_old = TT_old * TT_old
#         P_10 = Pt
#     end    #while diferenz
#     # println(P_10)
#     # Pt = copy(P_10)
#     # loglh = 0.0
#     # yaux = similar(DD)
#     # ZPZ = similar(HH)
#     # TTPt = similar(Pt)
#     # TAt = similar(At)
#     # KiF = similar(At)
#     # PtZZ = similar(Pt*ZZ')
#     # Kt = similar(Pt*ZZ')
#     @time begin
#     TTPtTT = copy(Pt)
#     KFK = copy(Pt)
#     # iFtnut = similar(DD)
    
#     yaux, yhat, TTPt, Kt, iFtnut = similar(DD), similar(DD), similar(Pt), similar(Pt*ZZ'), similar(DD)
#     Ft, iFt, nut = zeros(ny,ny), zeros(ny,ny), zeros(ny)
#     dFt = 0.0
#     end
#     # if isnan(sum(At))
#     #     println(At)
#     # end

#     @time @inbounds for i in 1:nobs
#     # @inbounds for i in 1:nobs
#         mul!(yaux,ZZ,At)
#         # println(DD)
#         # sleep(10)
#         yhat = yaux + DD
#         # if isnan(sum(yhat))
#         #     println(i)
#         # end
#         # println(y[i,:])
#         # println(yhat)
#         # sleep(10)
#         nut = y[i, :] - yhat
#         # println(nut)
#         # sleep(10)
#         # println(typeof(ZZ))
#         # println(typeof(Pt))
#         # println(typeof(PtZZ))
#         # @time @views mul!(PtZZ,Pt,ZZ')
#         # println(typeof(ZPZ))
#         # @time @views mul!(ZPZ,ZZ,PtZZ)
#         # @time @views ZPZ = ZZ * Pt * ZZ'
#         Ft = ZZ * Pt * ZZ' + HH #ZZ*Pt*ZZ' .+ HH
#         # println(Ft)
#         # sleep(10)
#         Ft = 0.5*(Ft + Ft')
#         # println(Ft)
#         # sleep(10)
#         dFt = det(Ft)
#         iFt = inv(Ft)
#         # println(Ft)
#         # sleep(10)
#         mul!(iFtnut,iFt,nut)
#         # println(iFtnut)
#         # sleep(10)
        
#         # if isnan(sum(nut*iFtnut))
#         #     println(i)
#         # end
#         loglhvec[i] = -0.5*(log(dFt) + (iFtnut'*nut))
#         # loglhvec[i] = -mean([log(dFt); (0.5*iFtnut'*nut)])
#         # println(loglhvec[i])
#         # sleep(10)
#         mul!(TTPt,TT,Pt)
#         mul!(Kt,TTPt,ZZ')
#         # @time @views TAt = TT * At
#         # @time @views KiF = Kt * iFtnut
#         # @time @views At = TAt + KiF
#         At = TT * At + Kt * iFtnut
#         mul!(TTPtTT,TTPt,TT')
#         mul!(KFK,Kt,(iFt*Kt'))
#         # KFK = Kt * iFt * Kt'
#         Pt = TTPtTT - KFK + RQR
#         # sleep(10)
#         # At, Pt, loglhvec[i] = kf_inner_loop(yaux,ZZ,At,DD, yhat, nut, Pt, Ft, HH, iFtnut, TTPt, TT, Kt, TTPtTT, KFK, iFt, RQR, y[i,:])
#     end
#     # loglh = sum(loglhvec) - nobs*0.5*ny*log(2.0*π)
#     # return loglh
#     # return sum(loglhvec) - nobs*0.5*ny*log(2.0*π)
#     return @time sum(loglhvec) - aux
# end    

# function kf_inner_loop(yaux,ZZ,At,DD, yhat, nut, Pt, Ft, HH, iFtnut, TTPt, TT, Kt, TTPtTT, KFK, iFt, RQR, y)
#     mul!(yaux,ZZ,At)
#     # println(DD)
#     # sleep(10)
#     yhat .= yaux + DD
#     # if isnan(sum(yhat))
#     #     println(i)
#     # end
#     # println(y[i,:])
#     # println(yhat)
#     # sleep(10)
#     nut .= y - yhat
#     # println(nut)
#     # sleep(10)
#     # println(typeof(ZZ))
#     # println(typeof(Pt))
#     # println(typeof(PtZZ))
#     # @time @views mul!(PtZZ,Pt,ZZ')
#     # println(typeof(ZPZ))
#     # @time @views mul!(ZPZ,ZZ,PtZZ)
#     # @time @views ZPZ = ZZ * Pt * ZZ'
#     Ft .= ZZ * Pt * ZZ' + HH #ZZ*Pt*ZZ' .+ HH
#     # println(Ft)
#     # sleep(10)
#     Ft .= 0.5*(Ft + Ft')
#     # println(Ft)
#     # sleep(10)
#     dFt = det(Ft)
#     iFt = inv(Ft)
#     # println(Ft)
#     # sleep(10)
#     mul!(iFtnut,iFt,nut)
#     # println(iFtnut)
#     # sleep(10)
    
#     # if isnan(sum(nut*iFtnut))
#     #     println(i)
#     # end
#     # loglhvec[i] = -0.5*(log(dFt) + (iFtnut'*nut))
#     # loglhvec[i] = -mean([log(dFt); (0.5*iFtnut'*nut)])
#     # println(loglhvec[i])
#     # sleep(10)
#     mul!(TTPt,TT,Pt)
#     mul!(Kt,TTPt,ZZ')
#     # @time @views TAt = TT * At
#     # @time @views KiF = Kt * iFtnut
#     # @time @views At = TAt + KiF
#     At = TT * At + Kt * iFtnut
#     mul!(TTPtTT,TTPt,TT')
#     mul!(KFK,Kt,(iFt*Kt'))
#     # KFK = Kt * iFt * Kt'
#     Pt = TTPtTT - KFK + RQR
#     return At, Pt, -0.5*(log(dFt) + (iFtnut'*nut))
# end

# function kf(y::Matrix{Float64}, TT::Matrix{Float64}, RQR::Matrix{Float64}, DD::Matrix{Float64}, ZZ::Matrix{Float64}, HH::Matrix{Float64}, t0::Int)
#     nobs, ny = size(y)
#     ns, ~ = size(TT)

#     At = zeros(Float64, ns, 1)
#     TT_old = TT
#     RQR_old = RQR
#     P_10_old = Matrix{Float64}(I, size(TT))
#     loglhvec = zeros(Float64, nobs)
#     P_10 = P_10_old
#     diferenz = 0.1

#     while diferenz > 1e-25
#         P_10 .= TT_old * P_10_old * TT_old' + RQR_old
#         diferenz = maximum(abs.(P_10 .- P_10_old))
#         RQR_old .= TT_old * RQR_old * TT_old' + RQR_old
#         TT_old .= TT_old * TT_old
#         P_10_old .= P_10
#     end

#     Pt = copy(P_10)
#     loglh = 0.0
#     yaux = similar(DD)
#     ZPZ = similar(HH)
#     TTPt = similar(Pt)
#     TAt = similar(At)
#     KiF = similar(At)
#     PtZZ = similar(Pt * ZZ')
#     Kt = similar(PtZZ)
#     TTPtTT = similar(Pt)
#     KFK = similar(Pt)
#     iFtnut = similar(DD)

#     if isnan(sum(At))
#         println(At)
#     end

#     for i in 1:nobs
#         mul!(yaux, ZZ, At)
#         yhat = yaux + DD
#         if isnan(sum(yhat))
#             println(i)
#         end

#         nut = (y[i, :] - yhat)'

#         mul!(PtZZ, Pt, ZZ')
#         mul!(ZPZ, ZZ, PtZZ)
#         Ft = ZPZ + HH
#         Ft = 0.5 * (Ft + Ft')

#         dFt = det(Ft)
#         iFt = inv(Ft)
#         mul!(iFtnut, iFt, nut')

#         if isnan(sum(nut * iFtnut))
#             println(i)
#         end

#         loglhvec[i] = -0.5 * log(dFt) - (0.5 * nut * iFtnut)[1]

#         mul!(TTPt, TT, Pt)
#         mul!(Kt, TTPt, ZZ')
#         mul!(TAt, TT, At)
#         mul!(KiF, Kt, iFtnut)
#         At .= TAt + KiF

#         mul!(TTPtTT, TTPt, TT')
#         mul!(KFK, Kt, (Ft \ Kt'))
#         Pt .= TTPtTT - KFK + RQR
#     end

#     loglh = sum(loglhvec) - nobs * 0.5 * ny * log(2 * π)
#     return loglh
# end

function logpriors(model::NamedTuple, p0)
    nparams=size(p0)
    idx=NaN*zeros(nparams)

    for i in 1:model.ns
        if p0[i] < model.priors[i].lb || p0[i] > model.priors[i].ub
            idx[i] = 0.0
        else
            idx[i] = pdf(model.priors[i].d, p0[i])
        end
    end

    lnp=sum(log.(idx))
    return lnp
end

function mutation_RWMH(model::NamedTuple, data::Matrix{Float64}, p0, l0::Float64, lpost0::Float64, c::Float64, R::Matrix, npara::Int, 
                    PAR::Vector{Float64}, PAR_SS::Vector{Float64}, VAR::Array{Float64}, SS::Vector{Float64}, deriv::NamedTuple, sol_mat::NamedTuple,
                    DD::Vector{Float64}, ZZ::Matrix{Float64}, HH::Matrix{Float64}, At::Vector{Float64}, Pt::Matrix{Float64}, loglhvec::Vector{Float64}, TTPtTT::Matrix{Float64},
                    KFK::Matrix{Float64}, aux::Float64)
    # RW proposal
    px = p0 .+ (c*cholesky(Hermitian(R)).U'*randn(npara))

    prior_val = logpriors(model, px)
    if prior_val == -Inf
        lnpost  = -Inf
        lnpY    = -Inf
        lnprio = -Inf
        error   = 10
    else
        lnpY = objfun(data, px, model, PAR, PAR_SS, VAR, SS, deriv, sol_mat, DD, ZZ, HH, At, Pt, loglhvec, TTPtTT, KFK, aux)
        lnpost = lnpY + prior_val
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

function smc_rwmh_neoclassical(model, data, PAR, c, npart, nphi, lam)
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

    @unpack flag_order, flag_deviation, parameters, x, y, xp, yp, variables = model
    # @unpack e, eta, f, n, nx, ny, ne, np, SS, skew = model
    @unpack e, eta, f, nf, nx, ny, ne, npar, SS = model

    # npara=size(initial_para,1)
    #do_geweke = false
    phi_bend = true
    # npart  = 500#2^13              # of particles
    # nphi   = 100 # 500         # of stage
    # lam    = 3#2.1                # bending coeff
    nobs = size(data,2)
    # println(nobs)
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

    parasim = initial_draw(model, npart, nphi)

    #priorsim       = priordraws[tune.npart]
    #parasim[1,:,:] = priorsim;          # from prior

    wtsim[:, 1]    = 1.0./npart .* ones(npart,1)      # initial weight is equal weights
    zhat[1]        = sum(wtsim[:,1])

    # Posterior values at prior draws
    loglh  = zeros(npart,1) #log-likelihood
    logpost = zeros(npart,1) #log-posterior
    #par
    @inbounds for i=1:npart
        #println(i)
        p0 = parasim[1,i,:]
        #println(size(p0))
        loglh[i] = objfun(data, p0, phi_smc, model, nobs, PAR)    
        prior_val = logpriors(model, p0)
        logpost[i]=loglh[i]+prior_val
        #logpost[if], loglh[i], lnprio[i], error[i] = objfun(data, p0, phi_smc, model, nobs)    
    end

    loglh[isnan.(loglh)] .= -1e50
    loglh[isinf.(loglh)] .= -1e50
    logpost[isnan.(logpost)] .= -1e50
    logpost[isinf.(logpost)] .= -1e50

    # ------------------------------------------------------------------------
    # Recursion: For n=2,...,N_[\phi]
    # ------------------------------------------------------------------------
    estimMean = zeros(nphi, model.ns)

    println("SMC recursion starts ... ")
    @inbounds for i=2:1:nphi
        
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
            @inbounds Threads.@threads for ii=1:npart   
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
        
        wght =  wtsim[:, i].*ones(npart,model.ns)
        
        mu      = sum(para.*wght) # mean()
        z       = (para - mu.*ones(npart, model.ns))
        R       = (z.*wght)'*z       # covariance
        Rdiag   = Diagonal(R)#diag(diag(R)) # covariance with diag elements
        Rchol   = cholesky(Hermitian(R)).U
        Rchol2  = sqrt(Rdiag)
        
        estimMean[i,:] .= mu
        
        # Particle mutation [Algorithm 2]
        temp_acpt = zeros(npart,1) #initialize accpetance indicator
        
        propmode=1
        #par
        @inbounds for j = 1:npart #iteration over particles
            # Options for proposals
            if propmode .== 1   
                # Mutation with RWMH
                ind_para, ind_loglh, ind_post, ind_acpt = mutation_RWMH(model, data, para[j,:]', loglh[j], logpost[j], c, R, model.ns, phi_smc, nobs, i, PAR)
                
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

######################################################################################################################################################################################
######################################################################################################################################################################################
function SS_solver(model, PAR)            
    f_aux = similar(model.f)
    copyto!(f_aux,model.f)
    for i in 1:model.nvar
        for iv in (model.nx+model.ny+1):(2*(model.nx+model.ny))
            f_aux[i] = f_aux[i].subs(model.variables[iv], model.variables[iv-model.nx-model.ny])
        end
    end
    for i in 1:model.nvar
        for ip in 1:model.npar
            f_aux[i] = f_aux[i].subs(model.parameters[ip], PAR[ip])
        end
    end
    f_aux = f_aux.subs(Dict(invefl => invef,
                            ccfl => ccf,
                            invel => inve,
                            yfl => yf,
                            cl => c,
                            pinfl => pinf, 
                            wl => w,
                            yl => yy,
                            rl => r
                            ))

    f_aux = f_aux.subs(Dict(a=>0, 
                            b=>0,
                            qs=>0,
                            ms=>0,
                            g=>0,
                            spinf=>0,
                            sw=>0,
                            epinfma=>0,
                            ewma=>0
                            ))
    
    # solu = solve(f_aux, model.variables[1:model.nx+model.ny])[1]
    solu = solve(simplify(f_aux), model.variables[1:model.nx+model.ny])
    # solu = nonlinsolve(simplify(f_aux))
    # i = 1
    SS = zeros(2*(model.nx+model.ny))
    for ii in 1:model.nx+model.ny
        if model.variables[ii] ∈ keys(solu)
            SS[ii] = solu[model.variables[ii]] 
            # println(solu[model.variables[ii]])
        end
    end
    return SS
end

######################################################################################################################################################################################
######################################################################################################################################################################################
