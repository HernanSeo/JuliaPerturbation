# (c) Alvaro Salazar-Perez and Hernán D. Seoane
# "Perturbating and Estimating DSGE models in Julia
# This version 2023

# If you use these codes, please cite us.
# These codes are free and without any guarantee. Use them at your own risk.
# If you find typos, please let us know.

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
function process_model(model::NamedTuple)        
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
    
    f_aux = similar(f)
    copyto!(f_aux,f)

    @inbounds for j in variables
        copyto!(f_aux,subs.(f_aux, j, (exp(j))))
    end

    @inbounds for iv in 1:2(nx+ny)
        copyto!(f_aux, f_aux.subs(variables[iv],Sym("SS["*string(iv)*"]")))
    end
    @inbounds for ip in 1:npar
        copyto!(f_aux, f_aux.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
    end
    
    SS_error_string = Meta.parse("function eval_SS_error(PAR, SS); return " * string(f_aux)[4:end] * "; end;")
    eval(SS_error_string)            

    if flag_deviation
        @inbounds for j in variables
            # if j ∉ [a, ap]
                copyto!(f,subs.(f, j, (exp(j))))
            # end
        end
    end
    
    fd_v = Array{Sym}(undef,0); fd_ci = Array{CartesianIndex}(undef,0)
    if flag_order > 1
        sd_v = Array{Sym}(undef,0); sd_ci = Array{CartesianIndex}(undef,0)
        if flag_order > 2
            td_v = Array{Sym}(undef,0); td_ci = Array{CartesianIndex}(undef,0)
        end
    end

    if flag_order < 3
        @inbounds for i in 1:nvar
            @inbounds for j in 1:2*(nx+ny)
                fd_aux = diff(f[i], variables[j])
                if fd_aux != 0
                    push!(fd_v, fd_aux)
                    push!(fd_ci, CartesianIndex(i,j))
                    if flag_order > 1
                        @inbounds for k in 1:2*(nx+ny)
                            sd_aux = diff(f[i], variables[j], variables[k])
                            if sd_aux != 0
                                push!(sd_v, sd_aux)
                                push!(sd_ci, CartesianIndex(i,j,k))
                                if flag_order > 2
                                    @inbounds for l in 1:2*(nx+ny)
                                        td_aux = diff(f[i], variables[j], variables[k], variables[l])
                                        if td_aux != 0
                                            push!(td_v, td_aux)
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
    else
        @inbounds for i in 1:nvar
            @inbounds for j in 1:2*(nx+ny)
                fd_aux = diff(f[i], variables[j])
                if fd_aux != 0
                    push!(fd_v, fd_aux)
                    push!(fd_ci, CartesianIndex(i,j))
                    @inbounds for k in 1:2*(nx+ny)
                        sd_aux = diff(f[i], variables[j], variables[k])
                        if sd_aux != 0
                            push!(sd_v, sd_aux)
                            push!(sd_ci, CartesianIndex(i,(j-1)*2(nx+ny)+k))
                            @inbounds for l in 1:2*(nx+ny)
                                td_aux = diff(f[i], variables[j], variables[k], variables[l])
                                if td_aux != 0
                                    push!(td_v, td_aux)
                                    push!(td_ci, CartesianIndex(i,(j-1)*4(nx+ny)^2+(k-1)*2(nx+ny)+l))
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
        deriv_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(model.nvar) * "; nx = " * string(model.nx) * "; ny = " * string(model.ny) * "; fd_ci = " * string(fd_ci) * "; sd_ci = " * string(sd_ci) * "; td_ci = " * string(td_ci) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(n, 4*(nx+ny)^2)), d3 = Array{Float64}(zeros(n, 8*(nx+ny)^3))); d1 = " * string(fd_v)[4:end] * "; @inbounds deriv.d1[fd_ci] = d1; d2 =" * string(sd_v)[4:end] * "; @inbounds deriv.d2[sd_ci] = d2; d3 =" * string(td_v)[4:end] * "; @inbounds deriv.d3[td_ci] = d3; return deriv; end;")    
    end
    eval(deriv_string)
end


function solve_model(model::NamedTuple, deriv::NamedTuple, eta)
    if model.flag_order < 2
        gx, hx, indic = first_order(model, deriv) 
        return (gx = gx, hx = hx, qzflag = indic)
    else
        if model.flag_order < 3
            gx, hx, indic, fx, fy, fxp, fyp = first_order(model, deriv) 
            if indic == 1
                gxx, hxx, gss, hss = second_order(model, deriv, eta, gx, hx, fx, fy, fxp, fyp)
            else
                gxx, hxx, gss, hss = [], [], [], [] 
            end
            return (gx = gx, hx = hx, qzflag = indic,
                    gxx = gxx, hxx = hxx, gss = gss, hss = hss)
        else
            gx, hx, indic, fx, fy, fxp, fyp = first_order(model, deriv) 
            if indic == 1
                gxx, hxx, gss, hss, gxxx, hxxx, gssx, hssx, gsss, hsss = third_order(model, deriv, eta, gx, hx, fx, fy, fxp, fyp)           
            else
                gxx, hxx, gss, hss, gxxx, hxxx, gssx, hssx, gsss, hsss = [], [], [], [], [], [], [], [], [], []
            end
            return (gx = gx, hx = hx, qzflag = indic,
                    gxx = gxx, hxx = hxx, gss = gss, hss = hss,
                    gxxx = gxxx, hxxx = hxxx, gssx = gssx, hssx = hssx, gsss = gsss, hsss = hsss)    
        end
    end
end


function first_order(model::NamedTuple, deriv::NamedTuple)
    @unpack parameters, estimate, estimation, npar, ns = model
    @unpack x, y, xp, yp, variables, nx, ny, nvar = model 
    
    fd_ = deriv.d1
    
    fx  = @views fd_[:,1:nx]
    fy  = @views fd_[:,nx+1:nvar]
    fxp = @views fd_[:,nvar+1:nvar+nx]
    fyp = @views fd_[:,nvar+nx+1:2*nvar]

    F = schur([-fxp -fyp],[fx  fy])

    slt = abs.(diag(F.T,0)).<abs.(diag(F.S,0))
    nk=sum(slt)

    F = ordschur!(F,slt)
    
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
        z11i = z11\Matrix(I, nk, nk)

        gx = real(z21*z11i)
        hx = real(z11*(s11\t11)*z11i)
    else
        gx = []
        hx = []
    end
    return gx, hx, indic, fx, fy, fxp, fyp
end


function second_order(model::NamedTuple, deriv::NamedTuple, eta, gx::Array, hx::Array, fx::SubArray, fy::SubArray, fxp::SubArray, fyp::SubArray)
    @unpack npar, ns = model
    @unpack nx, ny, nvar = model 
    @unpack nf = model
    
    fd_ = deriv.d1
    sd = deriv.d2
    
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

    A = kron(Matrix{Float64}(I, nx+ny, nx+ny),hx'*gx')*(nfypyp*gx*hx + nfypxp*hx + nfypy*gx + nfypx) + 
        kron(Matrix{Float64}(I, nx+ny, nx+ny),gx')    *(nfyyp *gx*hx + nfyxp *hx + nfyy *gx + nfyx ) +     
        kron(Matrix{Float64}(I, nx+ny, nx+ny),hx')    *(nfxpyp*gx*hx + nfxpxp*hx + nfxpy*gx + nfxpx) + 
                            (nfxyp *gx*hx + nfxxp *hx + nfxy *gx + nfxx);   
    B = kron(nfyp, hx') ; 
    C = kron(nfy, Matrix{Float64}(I, nx, nx))
    D = kron(nfyp*gx, Matrix{Float64}(I, nx, nx))+ kron(nfxp, Matrix{Float64}(I, nx, nx))

    Qq = -[ kron(hx',B)+kron(Matrix{Float64}(I, nx, nx),C) kron(Matrix{Float64}(I, nx, nx), D)] \ (A[:])

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

            # Third Term
            q[ii] = q[ii] + sum(diag((fypxp[ii,:,:]*eta)'*gx*eta));

            # Fourth Term
            q[ii] =  q[ii] + sum(diag(( reshape(fyp[ii,:]'*reshape(gxx,ny,nx^2),nx,nx)*eta)'*eta));

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

            # Third Term
            q[ii] = q[ii] + sum(diag([(fypxp[ii,:,:]*eta)'*gx*eta][:,:]));

            # Fourth Term
            q[ii] =  q[ii] + sum(diag([( reshape(fyp[ii,:]'*reshape(gxx,ny,nx^2),nx,nx)*eta)'*eta][:,:]));

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
    return gxx, hxx, gss, hss
end


function third_order(model::NamedTuple, deriv::NamedTuple, eta, gx, hx, fx::SubArray, fy::SubArray, fxp::SubArray, fyp::SubArray)
    @unpack npar, ns = model
    @unpack nx, ny, nvar = model 
    @unpack e, ne = model
    
    sd = deriv.d2
    td = deriv.d3
    
    VAR = eta*eta'
    skew = zeros(ne,ne^2)

    Mx = [I; gx; hx; gx*hx]

    A = [fxp+fyp*gx fy]
    B = [zeros(nvar,nx) fyp]

    D_aux = zeros(nx^2,nx+ny)
    @inbounds for i = 1:nx+ny
        @views D_aux[:,i] = vec(Mx'*reshape(sd'[:,i],2*(nx+ny),2*(nx+ny))*Mx)
    end
    D = -Matrix(D_aux')

    z = martin_van_loan(A,B,hx,D,1)
    hxx = z[1:nx,:]
    gxx = z[nx+1:nvar,:]


    Ns = [zeros(nvar,nx); I; gx]

    A = [fxp+fyp*gx fy+fyp]

    S_aux = VAR

    p1 = zeros(nx^2,(nx+ny))
    @inbounds for i = 1:nx+ny
        @views p1[:,i]  .= vec((Ns*S_aux)'*reshape(sd'[:,i],2*(nx+ny),2*(nx+ny))*Ns)
    end
    F1_aux = Matrix(p1')
    k1 = Int(round(sqrt(size(F1_aux)[2])))
    F1 = zeros(size(F1_aux)[1])
    @inbounds for i = 1:k1
        @views F1 .+= F1_aux[:,i+(i-1)*k1]
    end

    p2 = zeros(nx^2,ny)
    @inbounds for i = 1:ny
        @views p2[:,i] = vec(S_aux'*reshape(gxx'[:,i],nx,nx)*Matrix{Float64}(I,nx,nx))
    end
    F2_aux = Matrix(p2')
    k2 = Int(round(sqrt(size(F2_aux, 2))))
    F2 = zeros(size(F2_aux, 1))
    @inbounds for i = 1:k2
        @views F2 .+= F2_aux[:,i+(i-1)*k2]
    end

    F = -F1 .- fyp*F2

    z = A\F

    hss = reshape(z[1:nx,:],nx)
    gss = reshape(z[nx+1:nvar,:],ny)


    Mx = [I; gx; hx; gx*hx]

    A = [fxp+fyp*gx fy]
    B = [zeros(nvar,nx) fyp]

    Mxx = [zeros(nx,nx^2); gxx; hxx; gxx*kron(hx,hx) + gx*hxx]


    hxx_dagger = zeros(nx^2,nx^3)
    @inbounds for i = 1:nx
        @views hxx_dagger[:,(i-1)*nx^2+1:i*nx^2] .= kron(hx,hxx[:,(i-1)*nx+1:i*nx])
    end

    Aux = reshape([1:nx^3;],1,nx,nx,nx)
    Ix = Matrix{Float64}(I,nx^3,nx^3)
    omega = (reshape(Ix[:,PermutedDimsArray(Aux,[1,4,2,3])],nx^3,nx^3)
           + reshape(Ix[:,PermutedDimsArray(Aux,[1,2,4,3])],nx^3,nx^3)
           + reshape(Ix[:,PermutedDimsArray(Aux,[1,2,3,4])],nx^3,nx^3))

    p1 = zeros(nx^3,nx+ny)
    @inbounds for i = 1:nx+ny
        @views v_tilda = reshape(td'[:,i],4*(nx+ny)^2,2*(nx+ny))*Mx
        p2 = zeros(nx^2,nx)
        @inbounds for i = 1:nx
            @views p2[:,i] = vec(Mx'*reshape(v_tilda[:,i],2*(nx+ny),2*(nx+ny))*Mx)
        end
        p1[:,i] .= vec(Matrix(p2))
    end
    D1 = Matrix(p1')

    p = zeros(nx^3,nx+ny)
    @inbounds for i = 1:nx+ny
        @views p[:,i] = @views p[:,i] = vec(Mxx'*reshape(sd'[:,i],2*(nx+ny),2*(nx+ny))*Mx)
    end
    D2 = Matrix(p')

    D = (-D1 .- D2*omega .- fyp*gxx*(kron(hx,hxx) .+ kron(hxx,hx) .+ hxx_dagger))

    z = martin_van_loan(A,B,hx,D,2)
    hxxx = z[1:nx,:]
    gxxx = z[nx+1:nvar,:]


    A = [fxp+fyp*gx fy+fyp]
    Ns = [zeros(nvar,nx); I; gx]
    Nsx = [zeros(nvar+nx,nx^2); gxx*kron(hx,Matrix{Float64}(I,nx,nx))]
    Pss_aux2 = gxx*kron(Matrix{Float64}(I,nx,nx),VAR)
    k = Int(round(sqrt(size(Pss_aux2)[2])))
    Pss_aux = zeros(size(Pss_aux2)[1])
    @inbounds for i = 1:k
        @views Pss_aux += Pss_aux2[:,i+(i-1)*k]
    end
    Pss = [zeros(nx,1); gss; hss; gx*hss + Pss_aux + gss]

    S_aux = Ns*VAR

    p = zeros(nx^3,nx+ny)
    @inbounds for i = 1:nx+ny
        @views v_tilda = reshape(td'[:,i],4*(nx+ny)^2,2*(nx+ny))*Mx
        p2 = zeros(nx^2,nx)
        @inbounds for i = 1:nx
            @views p2[:,i] = vec(S_aux'*reshape(v_tilda[:,i],2*(nx+ny),2*(nx+ny))*Ns)
        end
        p[:,i] .= vec(p2)
    end
    F1_aux = Matrix(p')
    (n1,n2) = size(F1_aux)
    k = Int(round(n2^(1//3)))
    F1 = zeros(n1,k)
    @inbounds for j = 1:k
        @inbounds for i = 1:k
            @views F1[:,j] += F1_aux[:,(j-1)+i+(i-1)*k^2]
        end
    end

    p = zeros(nx^3,nx+ny)
    @inbounds for i = 1:nx+ny
        @views p[:,i] = vec(S_aux'*reshape(sd'[:,i],2*(nx+ny),2*(nx+ny))*Nsx)
    end
    F2_aux = Matrix(p')
    (n1,n2) = size(F2_aux)
    k = Int(round(n2^(1//3)))
    F2 = zeros(n1,k)
    @inbounds for j = 1:k
        @inbounds for i = 1:k
            @views F2[:,j] += F2_aux[:,(j-1)+i+(i-1)*k^2]
        end
    end

    p = zeros(nx,nx+ny)
    @inbounds for i = 1:nx+ny
        @views p[:,i] = vec(Pss[:,:]'*reshape(sd'[:,i],2*(nx+ny),2*(nx+ny))*Mx)
    end
    F3 = Matrix(p')

    F4_aux = gxxx*kron(hx,Matrix{Float64}(I,nx^2,nx^2))*kron(Matrix{Float64}(I,nx^2,nx^2),VAR)
    (n1,n2) = size(F4_aux)
    k = Int(round(n2^(1//3)))
    F4 = zeros(n1,k)
    @inbounds for j = 1:k
        @inbounds for i = 1:k
            @views F4[:,j] += F4_aux[:,(j-1)+i+(i-1)*k^2]
        end
    end

    p = zeros(nx, ny)
    @inbounds for i = 1:ny
        @views p[:,i] = vec(hss[:,:]'*reshape(gxx'[:,i],nx,nx)*hx)
    end
    F5 = Matrix(p')

    F = (-F1 - 2*F2 - F3 -fyp*(F4 + F5))

    F = A\F

    z = dsylvester(B,hx,F)
    hssx = z[1:nx,:]
    gssx = z[nx+1:nvar,:]


    hsss = zeros(nx)
    gsss = zeros(ny)

    if sum(isequal.(skew,0.0)) == 0

        skew = zeros(nx,nx^2)
        skew[1:ns,1:ns^2] = skew

        Nss = [zeros(nv+nx,nx^2); gxx]

        F1_aux = td*kron(Ns,kron(Ns,Ns*skew))
            (n1,n2) = size(F1_aux)
            k = Int(round(sqrt(n2)))
            F1 = zeros(n1)
            @inbounds for i = 1:k
                @views F1 += F1_aux[:,i+(i-1)*k]
            end

        F2_aux = sd*kron(Nss,Ns*skew)
            (n1,n2) = size(F2_aux)
            k = Int(round(sqrt(n2)))
            F2 = zeros(n1)
            @inbounds for i = 1:k
                @views F2 += F2_aux[:,i+(i-1)*k]
            end

        F3_Aux = gxxx*kron(matrix(I,nx^2,nx^2),skew)
            (n1,n2) = size(F3_aux)
            k = Int(round(sqrt(n2)))
            F3 = zeros(n1)
            @inbounds for i = 1:k
                @views F3 += F3_aux[:,i+(i-1)*k]
            end

        F = -F1 - 3*F2 - fyp*F3

        z = A\F
        hsss = reshape(z[1:nx,:],nx)
        gsss = reshape(z[nx+1:nvar,:],ny)
    end

    return gxx, hxx, gss, hss, 
            gxxx, hxxx, gssx, hssx, gsss, hsss
end


function martin_van_loan(a::Matrix{Float64}, b::Matrix{Float64}, c::Matrix{Float64}, d::Matrix{Float64}, k::Int64)

    #Based on Martin and Van Loan (2006)

    a = copy(a)
    b = a\copy(b)
    c = copy(c)
    d = a\copy(d)

    (v,s) = hessenberg(b)       # v*s*v' = b
    (t,q) = schur(complex(c'))  # q*t*q' = c'

    v = Matrix{Float64}(v)

    p = k + 2
    TT = Array{typeof(t)}(undef,p)
    TT[1] = s
    @inbounds for i = 2:p
        TT[i] = conj(t)
    end

    Q     = fill(q,k+1)
    inv_Q = fill(Matrix(q'),k+1)

    n = fill(size(c,1),p)
    n[1] = size(b,1)
    N = prod(n)

    lambda = 1.0

    N = prod(n[2:end][1:k+1])
    e_aux = Array{Complex{Real}}(undef,N,size(Matrix(d'),2))
    @inbounds for j = 1:size(Matrix(d'),2)
        @views z = Matrix(d')[:,j]
        @inbounds for i = 1:k+1
            z = (inv_Q[i]*reshape(z,n[2:end][i],Int(N/n[2:end][i])))'
        end
        e_aux[:,j] .= reshape(Matrix(z),N)
    end
    e = Array{Complex}((v'*e_aux')[:])

    y = reshape(KPShiftSolve(TT,n,e,lambda,1.0),size(d))

    N = prod(n[2:end][1:k+1])
    x_aux = Array{Complex{Real}}(undef,N,size(Matrix(y'),2))
    @inbounds for j = 1:size(Matrix(y'),2)
        @views z = Matrix(y')[:,j]
        @inbounds for i = 1:k+1
            z = (Q[i]*reshape(z,n[2:end][i],Int(N/n[2:end][i])))'
        end
        x_aux[:,j] .= reshape(Matrix(z),N)
    end
    x = real(Matrix(v*x_aux'))

    return x
end


function dsylvester(a::Matrix{Float64}, b::Matrix{Float64}, c::Matrix{Float64})

    # Based on Golub, Nash, and Van Loan (1979).

    n = size(a, 1)
    m = size(b, 1)
    x = zeros(size(c))

    (s,u) = schur(Matrix(b'))
    (v,t) = hessenberg(a)

    c = v'*c*u

    j = m
    while j > 0
        j1 = j
        if j == 1
            block_size = 1
        elseif isequal(s[j,j-1],0.0) == false
            block_size = 2
            j -= 1
        else;
            block_size = 1
        end
        @views ajj = kron(s[j:j1,j:j1],t) + I
        @views rhs = vec(c[:,j:j1])
        if j1 < m
            @views rhs2 = t*(x[:,(j+1):m]*s[j:j1,(j+1):m]')
            rhs -= vec(rhs2)
        end
        w = ajj\rhs
        @views x[:,j] = w[1:n]
        if block_size == 2
            @views x[:,j1] = w[(n+1):2*n]
        end
        j -= 1
    end
    return v*x*u'
end


function KPShiftSolve(TT::Vector{Matrix{ComplexF64}}, n::Vector{Int64},c::Vector{Complex},
                      lambda::Float64, alpha)

    p = length(n)
    N = prod(n)

    c = copy(c)

    TT[p] = alpha*TT[p]
    if p == 1
        y = (TT[1] + lambda*Matrix{Real}(I,n[1],n[1]))\c
    else
        y = Array{Complex{Real}}(undef,N)
        mp = Int(N/n[p])
        @inbounds for i = n[p]:-1:1
            idx = ((i-1)*mp+1):(i*mp)
            y[idx] = KPShiftSolve(TT[1:(p-1)],n[1:(p-1)],c[idx],lambda,TT[p][i,i])

            N = prod(n[1:p-1])
            z = Array{Complex{Real}}(undef,N,size(y[idx],2))
            @inbounds for j = 1:size(y[idx],2)
                @views m = y[idx][:,j]
                @inbounds for i = 1:p-1
                    m = (TT[1:p][i]*reshape(m,n[i],Int(N/n[i])))'
                end
                z[:,j] .= reshape(Matrix(m),N)
            end

            @inbounds for j = 1:(i-1)
                jdx = ((j-1)*mp+1):(j*mp)
                c[jdx] = c[jdx] - TT[p][j,i]*z
            end
        end
    end

    return y
end


function simulate_model(model::NamedTuple, sol_mat::NamedTuple, TS::Int, eta::Array, SS::Vector, flag_IR::Bool, flag_logdev::Bool)
    @unpack nx, ny, ne, nvar = model
    @unpack flag_order, flag_deviation = model
    Random.seed!(1)

    if flag_IR
        sim_shocks = zeros(TS, ne)
        sim_shocks[1,:] = ones(ne)
    else
        sim_shocks = randn(TS, ne)
    end

    if flag_order == 1    
        sim_x = zeros(TS, nx)
        sim_y = zeros(TS, ny)
        @inbounds for t in 1:TS
            @inbounds for i in 1:ny
                sim_y[t, i] = sol_mat.gx[i, :]' * sim_x[t, :]
            end
            if t < TS
                @inbounds for i in 1:nx
                    sim_x[t+1,i] = sol_mat.hx[i,:]' * sim_x[t,:] + eta[i,:]'*sim_shocks[t,:]
                end
            end
        end

    elseif flag_order == 2
        sim_x_f = zeros(TS, nx)
        sim_x_s = zeros(TS, nx)
        sim_y = zeros(TS, ny)
        @inbounds for t in 1:TS
            @inbounds for i in 1:ny
                sim_y[t,i] = sol_mat.gx[i,:]' * (sim_x_f[t,:]+sim_x_s[t,:]) + 1/2 * sim_x_f[t,:]' * sol_mat.gxx[i,:,:] * sim_x_f[t,:] + 1/2 * sol_mat.gss[i] 
            end
            if t < TS
                @inbounds for i in 1:nx
                    sim_x_f[t+1,i] = sol_mat.hx[i,:]' * sim_x_f[t,:] + eta[i,:]'*sim_shocks[t,:]
                    sim_x_s[t+1,i] = sol_mat.hx[i,:]' * sim_x_s[t,:] + 1/2 * sim_x_f[t,:]' * sol_mat.hxx[i,:,:] * sim_x_f[t,:] + 1/2 * sol_mat.hss[i]
                end
            end
        end
        sim_x = sim_x_f + sim_x_s

    elseif flag_order == 3
        sim_x_f = zeros(TS, nx)
        sim_y_f = zeros(TS, ny)
        sim_x_s = zeros(TS, nx)
        sim_y_s = zeros(TS, ny)
        sim_x_t = zeros(TS, nx)
        sim_y_t = zeros(TS, ny)

        @inbounds for t in 1:TS
            sim_y_f[t,:] = sol_mat.gx*sim_x_f[t,:]
            sim_y_s[t,:] = sol_mat.gx*sim_x_s[t,:] + (1/2)*sol_mat.gxx*kron(sim_x_f[t,:],sim_x_f[t,:]) + (1/2)*sol_mat.gss
            sim_y_t[t,:] = sol_mat.gx*sim_x_t[t,:] + (1/2)*sol_mat.gxx*kron(sim_x_f[t,:],sim_x_s[t,:]) + (1/6)*sol_mat.gxxx*kron(kron(sim_x_f[t,:],sim_x_f[t,:]),sim_x_f[t,:]) + (3/6)*sol_mat.gssx*sim_x_f[t,:] + (1/6)*sol_mat.gsss
            if t < TS
                if ne > 1
                    sim_x_f[t+1,:]  = sol_mat.hx*sim_x_f[t,:] + eta*sim_shocks[t,:]
                else    
                    sim_x_f[t+1,:]  = sol_mat.hx*sim_x_f[t,:] + eta*sim_shocks[t,1]
                end
                sim_x_s[t+1,:]  = sol_mat.hx*sim_x_s[t,:] + (1/2)*sol_mat.hxx*kron(sim_x_f[t,:],sim_x_f[t,:]) + (1/2)*sol_mat.hss
                sim_x_t[t+1,:]  = sol_mat.hx*sim_x_t[t,:] + (1/2)*sol_mat.hxx*kron(sim_x_f[t,:],sim_x_s[t,:]) + (1/6)*sol_mat.hxxx*kron(kron(sim_x_f[t,:],sim_x_f[t,:]),sim_x_f[t,:]) + (3/6)*sol_mat.hssx*sim_x_f[t,:] + (1/6)*sol_mat.hsss
            end
        end        
        sim_x = sim_x_f + sim_x_s + sim_x_t
        sim_y = sim_y_f + sim_y_s + sim_y_t
    end

    if flag_logdev
        return [sim_x sim_y]
    else
        if flag_deviation
            return exp.([sim_x sim_y]).*repeat(exp.(SS[1:nvar]'), TS, 1)
        else
            return SS[1:nvar]' .+ [sim_x sim_y]
        end
    end

end


function smc_rwmh(model::NamedTuple, data::Matrix{Float64}, PAR::Vector{Float64}, PAR_SS::Vector{Float64}, VAR::Array{Float64}, SS::Vector{Float64}, 
                    deriv::NamedTuple, sol_mat::NamedTuple, c::Float64, npart::Int, nphi::Int, lam::Int)
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


    phi_bend = true
    ny, nobs = size(data)
    acpt=0.25
    trgt=0.25
    
    DD = zeros(nobs)
    ZZ = Matrix{Float64}(I, nobs, model.ny)
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
                id[ii]=sample(1:npart, AnalyticWeights(wtsim[:,i]))
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


function smc_rwmh_threads(model::NamedTuple, data::Matrix{Float64}, PAR::Vector{Float64}, PAR_SS::Vector{Float64}, VAR::Array{Float64}, SS::Vector{Float64}, 
                            deriv::NamedTuple, sol_mat::NamedTuple, c::Float64, npart::Int, nphi::Int, lam::Int)
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

    phi_bend = true
    ny, nobs = size(data)
    acpt=0.25
    trgt=0.25
    
    DD = zeros(nobs)
    ZZ = Matrix{Float64}(I, nobs, model.ny)
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
                id[ii]=sample(1:npart, AnalyticWeights(wtsim[:,i]))
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
    if !model.flag_SSsolver
        SS  =   eval_SS(PAR_SS)
    else
        SS  =   SS_solver(model, PAR)
    end
    deriv   =   eval_deriv(PAR_SS, SS)

    sol_mat     =   solve_model(model, deriv, VAR)

    if sol_mat.qzflag .== 1
        return kf(yy,
                    sol_mat.hx,
                    VAR*VAR',
                    DD,
                    ZZ*sol_mat.gx,
                    HH,
                    At, Pt, loglhvec, TTPtTT, KFK, aux)
    else
        return -1e16
    end
end


function kf(y::Matrix{Float64}, TT::Matrix{Float64}, RQR::Matrix{Float64}, DD::Array{Float64}, ZZ::Matrix{Float64}, HH::Matrix{Float64}, 
            At::Vector, Pt::Matrix, loglhvec::Vector{Float64}, TTPtTT::Matrix{Float64}, KFK::Matrix{Float64}, aux::Float64)
    nobs, ny = size(y)
    
    TT_old = copy(TT)
    RQR_old = copy(RQR)
    P_10 = copy(Pt)
    diferenz=0.1

    while diferenz>1e-25
        Pt = TT_old*P_10*TT_old' 
        Pt += RQR_old
        diferenz = maximum(abs.(Pt-P_10))
        RQR_old = TT_old*RQR_old*TT_old' + RQR_old
        TT_old = TT_old * TT_old
        P_10 = Pt
    end    #while diferenz

    TTPtTT = copy(Pt)
    KFK = copy(Pt)
    
    yaux, yhat, TTPt, Kt, iFtnut = similar(DD), similar(DD), similar(Pt), similar(Pt*ZZ'), similar(DD)
    Ft, iFt, nut = zeros(ny,ny), zeros(ny,ny), zeros(ny)
    dFt = 0.0

    @inbounds for i in 1:nobs
        mul!(yaux,ZZ,At)
        yhat = yaux + DD
        nut = y[i, :] - yhat
        Ft = ZZ * Pt * ZZ' + HH
        Ft = 0.5*(Ft + Ft')
        dFt = det(Ft)
        iFt = inv(Ft)
        mul!(iFtnut,iFt,nut)
        loglhvec[i] = -0.5*(log(dFt) + (iFtnut'*nut))
        mul!(TTPt,TT,Pt)
        mul!(Kt,TTPt,ZZ')
        At = TT * At + Kt * iFtnut
        mul!(TTPtTT,TTPt,TT')
        mul!(KFK,Kt,(iFt*Kt'))
        Pt = TTPtTT - KFK + RQR
    end
    sum(loglhvec) - aux
end    


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
