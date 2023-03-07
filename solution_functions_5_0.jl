# (c) Alvaro Salazar-Perez and Hernán D. Seoane
# "Perturbating and Estimating DSGE models in Julia
# This version 2023

# If you use these codes, please cite us.
# These codes are free and without any guarantee. Use them at your own risk.
# If you find typos, please let us know.

## Functions

    function create_model(flag_order::Int64, flag_deviation::Bool,
                                parameters::Vector{Sym}, estimate::Any,
                                x::Vector{Sym}, y::Vector{Sym}, xp::Vector{Sym}, yp::Vector{Sym},
                                e::Vector{Sym}, eta::Any,
                                f::Vector{Sym},
                                SS::Vector{Sym}, PAR_SS::Vector{Sym})
        variables = [x; y; xp; yp]

        if isempty(SS)
            flag_SSestimation = true
        else
            flag_SSestimation = false
        end

        flag_nologlin = false
        vec_nologlin = Array{Sym}(undef, 0)

        n = size(f,1); nx = size(x,1); ny = size(y,1); ne = size(e,1); np = size(parameters,1);
        if isempty(estimate)
            return (flag_order = flag_order, flag_deviation = flag_deviation,
                    flag_SSestimation = flag_SSestimation, flag_nologlin = flag_nologlin,
                    vec_nologlin = vec_nologlin,
                    parameters = parameters, x = x, y = y, xp = xp, yp = yp, variables = variables,
                    e = e, eta = eta, f = f, n = n, nx = nx, ny = ny, ne = ne, np = np,
                    SS = SS, PAR_SS = PAR_SS, skew = zeros(ne,ne^2))
        else
            estimation = []
            @inbounds for ii in 1:size(estimate,1)
                @inbounds for jj in 1:np
                    if parameters[jj] == estimate[ii]
                        push!(estimation,jj)
                    end
                end
            end

            # ns =

            return (flag_order = flag_order, flag_deviation = flag_deviation,
                    flag_SSestimation = flag_SSestimation, flag_nologlin = flag_nologlin,
                    vec_nologlin = vec_nologlin,
                    parameters = parameters, estimation = estimation,
                    x = x, y = y, xp = xp, yp = yp, variables = variables,
                    e = e, eta = eta, f = f, n = n, nx = nx, ny = ny, ne = ne, np = np, ns = length(estimation),
                    SS = SS, PAR_SS = PAR_SS, skew = zeros(ne,ne^2))
        end
    end

#function steadystate(f, variables, n, nx, ny, deviation, SSestimation; SS = Array{Sym}(undef, 2(nx+ny)))
    function steadystate(model::NamedTuple)
        @unpack flag_order, flag_deviation, flag_SSestimation, parameters, x, y, xp, yp, variables = model
        @unpack e, eta, f, n, nx, ny, ne, np, SS, skew = model

        # flag = (value = false, no_vec = Array{Sym}(undef, 0))

        if flag_SSestimation
            fss = copy(f)

            @inbounds for iv in 1:(nx+ny)
                fss = simplify.(subs.(fss, variables[(nx+ny)+iv],variables[iv]))
            end

            SS_aux = SymPy.solve(fss, variables[1:(nx+ny)])
            # SS = Array{Sym}(undef, 2(nx+ny))

            if flag_deviation
                @inbounds for iv in 1:(nx+ny)
                    if SS_aux[1][iv] == 0
                        copyto!(model.flag_nologlin, true)
                        model.vec_nologlin = push!(model.vec_nologlin, variables[iv])
                        SS[iv] = SS_aux[1][iv]
                        SS[(nx+ny)+iv] = SS_aux[1][iv]
                    else
                        SS[iv] = log(SS_aux[1][iv])
                        SS[(nx+ny)+iv] = log(SS_aux[1][iv])
                    end
                end
            else
                @inbounds for iv in 1:(nx+ny)
                    SS[iv] = SS_aux[1][iv]
                    SS[(nx+ny)+iv] = SS_aux[1][iv]
                end
            end
        end

        @inbounds for ip in 1:np
            copyto!(SS, SS.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
        end

        function_string = Meta.parse("function eval_SS(PAR); return " * string(SS)[4:end] * "; end;"
        )

        return function_string
    end

    function adjustpar(model::NamedTuple)
        @unpack flag_order, flag_deviation, flag_SSestimation, parameters, x, y, xp, yp, variables = model
        @unpack e, eta, f, n, nx, ny, ne, np, SS, PAR_SS, skew = model


        @inbounds for ip in 1:np
            copyto!(PAR_SS, PAR_SS.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
        end

        function_string = Meta.parse("function eval_PAR_SS(PAR); return " * string(PAR_SS)[4:end] * "; end;"
                                    )

        return function_string
    end

    function ss_error(model::NamedTuple)
        @unpack flag_order, flag_deviation, flag_SSestimation, parameters, x, y, xp, yp, variables = model
        @unpack e, eta, f, n, nx, ny, ne, np, SS, PAR_SS, skew = model

        f_aux = similar(f)
        copyto!(f_aux,f)
        
        @inbounds for j in variables
            copyto!(f_aux,subs.(f_aux, j, (exp(j))))
        end

        @inbounds for iv in 1:2(nx+ny)
            copyto!(f_aux, f_aux.subs(variables[iv],Sym("SS["*string(iv)*"]")))
        end
        @inbounds for ip in 1:np
            copyto!(f_aux, f_aux.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
        end
        
        function_string = Meta.parse("function eval_SS_error(PAR, SS); return " * string(f_aux)[4:end] * "; end;"
                                    )

        return function_string
    end


function ShockVAR(model::NamedTuple)
    @unpack np, eta, parameters = model

    eta_aux = eta
    neta = length(eta)

    if typeof(eta) == Vector{Sym} || typeof(eta) == Matrix{Sym}
        @inbounds for i1 in 1:size(eta_aux,1)
            @inbounds for i2 in 1:size(eta_aux,2)
                @inbounds for ip in 1:model.np
                    eta_aux[i1,i2] = subs(eta_aux[i1,i2], parameters[ip], Sym("PAR["*string(ip)*"]"))
                end
            end
        end
        function_string = Meta.parse("function eval_ShockVAR(PAR); VAR = Array{Float64}(zeros("*string(neta)*","*string(neta)*")); VAR = "*string(eta_aux)[4:end]*"*"*string(eta_aux)[4:end]*"'; return VAR; end")
    else
        function_string = Meta.parse("function eval_ShockVAR(PAR); return "*string(eta_aux)*"*"*string(eta_aux)*"'; end")
    end

    return function_string
end


function derivatives(model::NamedTuple)
    @unpack flag_order, flag_deviation, flag_SSestimation,  parameters, x, y, xp, yp, variables = model
    @unpack flag_nologlin, vec_nologlin = model
    @unpack e, eta, f, n, nx, ny, ne, np, SS, skew = model

    if flag_deviation
        if flag_nologlin
            @inbounds for j in variables
               if j ∉ vec_nologlin
                   copyto!(f,subs.(f, j, (exp(j))))
               end
           end
       else
        @inbounds for j in variables
               copyto!(f,subs.(f, j, (exp(j))))
           end
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

    @inbounds for i in 1:n
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
                            push!(sd_ci, CartesianIndex(i,(j-1)*2(nx+ny)+k))
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
                                        push!(td_ci, CartesianIndex(i,(j-1)*4(nx+ny)^2+(k-1)*2(nx+ny)+l))
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
        @inbounds for ip in 1:np
            copyto!(fd_v, fd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
        end
        # function_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(model.n) * "; nx = " * string(model.nx) * "; ny = " * string(model.ny) * "; fd_l = " * string(fd_l) * "; fd_c = " * string(fd_c) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(0)), d3 = Array{Float64}(zeros(0))); d1 = " * string(fd_v)[4:end] * "; @inbounds for ii in 1:length(d1); deriv.d1[fd_l[ii], fd_c[ii]] = d1[ii]; end; return deriv; end;")
        function_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(model.n) * "; nx = " * string(model.nx) * "; ny = " * string(model.ny) * "; fd_ci = " * string(fd_ci) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(0)), d3 = Array{Float64}(zeros(0))); d1 = " * string(fd_v)[4:end] * "; @inbounds deriv.d1[fd_ci] = d1; return deriv; end;")    
        return function_string
    elseif flag_order == 2
        @inbounds for iv in 1:2(nx+ny)
            copyto!(fd_v, fd_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
            copyto!(sd_v, sd_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
        end
        @inbounds for ip in 1:np
            copyto!(fd_v, fd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
            copyto!(sd_v, sd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
        end
        # function_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(model.n) * "; nx = " * string(model.nx) * "; ny = " * string(model.ny) * "; fd_l = " * string(fd_l) * "; fd_c = " * string(fd_c) * "; sd_l = " * string(sd_l) * "; sd_c = " * string(sd_c) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(n, 4*(nx+ny)^2)), d3 = Array{Float64}(zeros(0))); d1 = " * string(fd_v)[4:end] * "; for ii in 1:length(d1); deriv.d1[fd_l[ii],fd_c[ii]] = d1[ii]; end; d2 =" * string(sd_v)[4:end] * "; @inbounds for ii in 1:length(d2); deriv.d2[sd_l[ii],sd_c[ii]] = d2[ii]; end; return deriv; end;")
        function_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(model.n) * "; nx = " * string(model.nx) * "; ny = " * string(model.ny) * "; fd_ci = " * string(fd_ci) * "; sd_ci = " * string(sd_ci) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(n, 4*(nx+ny)^2)), d3 = Array{Float64}(zeros(0))); d1 = " * string(fd_v)[4:end] * "; @inbounds deriv.d1[fd_ci] = d1; d2 =" * string(sd_v)[4:end] * "; @inbounds deriv.d2[sd_ci] = d2; return deriv; end;")
        return function_string
    elseif flag_order == 3
        @inbounds for iv in 1:2(nx+ny)
            copyto!(fd_v, fd_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
            copyto!(sd_v, sd_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
            copyto!(td_v, td_v.subs(variables[iv],Sym("SS["*string(iv)*"]")))
        end
        @inbounds for ip in 1:np
            copyto!(fd_v, fd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
            copyto!(sd_v, sd_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
            copyto!(td_v, td_v.subs(parameters[ip],Sym("PAR["*string(ip)*"]")))
        end
        # function_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(model.n) * "; nx = " * string(model.nx) * "; ny = " * string(model.ny) * "; fd_l = " * string(fd_l) * "; fd_c = " * string(fd_c) * "; sd_l = " * string(sd_l) * "; sd_c = " * string(sd_c) * "; td_l = " * string(td_l) * "; td_c = " * string(td_c) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(n, 4*(nx+ny)^2)), d3 = Array{Float64}(zeros(n, 8*(nx+ny)^3))); d1 = " * string(fd_v)[4:end] * "; for ii in 1:length(d1); deriv.d1[fd_l[ii],fd_c[ii]] = d1[ii]; end; d2 =" * string(sd_v)[4:end] * "; for ii in 1:length(d2); deriv.d2[sd_l[ii],sd_c[ii]] = d2[ii]; end; d3 =" * string(td_v)[4:end] * "; @inbounds for ii in 1:length(d3); deriv.d3[td_l[ii],td_c[ii]] = d3[ii]; end; return deriv; end;")
        function_string = Meta.parse("function eval_deriv(PAR, SS); n = " * string(model.n) * "; nx = " * string(model.nx) * "; ny = " * string(model.ny) * "; fd_ci = " * string(fd_ci) * "; sd_ci = " * string(sd_ci) * "; td_ci = " * string(td_ci) * "; deriv = (d1 = Array{Float64}(zeros(n, 2*(nx+ny))), d2 = Array{Float64}(zeros(n, 4*(nx+ny)^2)), d3 = Array{Float64}(zeros(n, 8*(nx+ny)^3))); d1 = " * string(fd_v)[4:end] * "; @inbounds deriv.d1[fd_ci] = d1; d2 =" * string(sd_v)[4:end] * "; @inbounds deriv.d2[sd_ci] = d2; d3 =" * string(td_v)[4:end] * "; @inbounds deriv.d3[td_ci] = d3; return deriv; end;")    
        return function_string
    end
end


function solve_perturbation(model::NamedTuple, deriv::NamedTuple, VAR::Array{Float64,2})
    @unpack flag_order, flag_deviation, flag_SSestimation,  parameters, x, y, xp, yp, variables = model
    @unpack flag_nologlin, vec_nologlin = model
    @unpack e, eta, f, n, nx, ny, ne, np, SS, skew = model


    if flag_order == 1
        fo_mat = first_order(deriv.d1, n, nx, ny)   # (gx, hx)
        return (fo_mat = fo_mat, place_holder = zeros(2))
    elseif flag_order == 2
        fo_mat = first_order(deriv.d1, n, nx, ny)   # (gx, hx)
        so_mat = second_order(deriv.d1, deriv.d2, VAR, fo_mat, n, nx, ny)   # (gxx, hxx, gss, hss)
        return (fo_mat = fo_mat, so_mat = so_mat)
    else
        fo_mat = first_order(deriv.d1, n, nx, ny)   # (gx, hx)
        so_mat = second_order(deriv.d1, deriv.d2, VAR, fo_mat, n, nx, ny)   # (gxx, hxx, gss, hss)
        to_mat = third_order(deriv.d1, deriv.d2, deriv.d3, VAR, skew, fo_mat, so_mat, n, nx, ny)   # (gxxx, hxxx, gssx, hssx, gsss, hsss)
        return (fo_mat = fo_mat, so_mat = so_mat, to_mat = to_mat)
    end
end


function first_order(fd::Matrix{Float64}, n::Int64, nx::Int64, ny::Int64)

    @views fx  = fd[:,1:nx]
    @views fy  = fd[:,nx+1:n]
    @views fxp = fd[:,n+1:n+nx]
    @views fyp = fd[:,n+nx+1:2*n]

    # Complex Schur Decomposition
    F = schur([-fxp -fyp],[fx  fy])

    # Pick non-explosive (stable) eigenvalues
    slt = abs.(diag(F.T,0)).<abs.(diag(F.S,0))
    nk=sum(slt)

    # Reorder the system with stable eigs in upper-left
    F = ordschur!(F,slt)

    # Split up the results appropriately
    z21 = F.Z[nk+1:end,1:nk]
    z11 = F.Z[1:nk,1:nk]

    s11 = F.S[1:nk,1:nk]
    t11 = F.T[1:nk,1:nk]

    indic = 1 # indicates that equilibrium exists and is unique

    # Identify cases with no/multiple solutions
    if nk > size(fx,2)
        println("The Equilibrium is Locally Indeterminate")
        indic = 2
    elseif nk < size(fx,2)
        println("No Local Equilibrium Exists")
        indic = 0
    end

    if rank(z11) < nk
        println("Invertibility condition violated")
        indic = 3
    end

    # Compute the Solution
    z11i = z11\Matrix(I, nk, nk)

    return (gx = real(z21*z11i), hx = real(z11*(s11\t11)*z11i), qzflag = indic)   # (gx, hx)
end


function second_order(fd::Matrix{Float64}, sd::Matrix{Float64},
                      VAR::Array{Float64,2}, fo_mat::NamedTuple,
                      n::Int64, nx::Int64, ny::Int64)

    gx = fo_mat.gx
    hx = fo_mat.hx

    # @views fx  = fd[:,1:nx]
    # @views fy  = fd[:,nx+1:n]
    # @views fxp = fd[:,n+1:n+nx]
    # @views fyp = fd[:,n+nx+1:2*n]

    fx  = view(fd,:,1:nx)
    fy  = view(fd,:,nx+1:n)
    fxp = view(fd,:,n+1:n+nx)
    fyp = view(fd,:,n+nx+1:2*n)

    Mx = [I; gx; hx; gx*hx]

    A = [fxp+fyp*gx fy]
    B = [zeros(n,nx) fyp]

    D_aux = zeros(nx^2,nx+ny)
    @inbounds for i = 1:nx+ny
        @views D_aux[:,i] = vec(Mx'*reshape(sd'[:,i],2*(nx+ny),2*(nx+ny))*Mx)
    end
    D = -Matrix(D_aux')

    # D_aux = zeros(nx^2,nx+ny)
    # for i = 1:nx+ny
    #     D_aux[:,i] .= vec(Mx'*reshape(sd'[:,i],2*(nx+ny),2*(nx+ny))*Mx)
    # end
    # D = -Matrix(D_aux')

    z = martin_van_loan(A,B,hx,D,1)
    hxx = z[1:nx,:]
    gxx = z[nx+1:n,:]



    Ns = [zeros(n,nx); I; gx]

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
    gss = reshape(z[nx+1:n,:],ny)

    return (gxx = gxx, hxx = hxx,
            gss = gss, hss = hss)
end


function third_order(fd::Matrix{Float64}, sd::Matrix{Float64}, td::Matrix{Float64},
                        VAR::Array{Float64,2}, skew::Matrix{Float64},
                        fo_mat::NamedTuple, so_mat::NamedTuple,
                        n::Int64, nx::Int64, ny::Int64)

    @views fx  = fd[:,1:nx]
    @views fy  = fd[:,nx+1:n]
    @views fxp = fd[:,n+1:n+nx]
    @views fyp = fd[:,n+nx+1:2*n]

    gx = fo_mat.gx
    hx = fo_mat.hx
    gxx = so_mat.gxx
    hxx = so_mat.hxx
    gss = so_mat.gss
    hss = so_mat.hss

    Mx = [I; gx; hx; gx*hx]

    A = [fxp+fyp*gx fy]
    B = [zeros(n,nx) fyp]

    Mxx = [zeros(nx,nx^2); gxx; hxx; gxx*kron(hx,hx) + gx*hxx]

    # Construct the hxx_dagger matrix

    hxx_dagger = zeros(nx^2,nx^3)
    @inbounds for i = 1:nx
        @views hxx_dagger[:,(i-1)*nx^2+1:i*nx^2] .= kron(hx,hxx[:,(i-1)*nx+1:i*nx])
    end

    # Solve for the third order coefficients on the state variables
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
    gxxx = z[nx+1:n,:]



    A = [fxp+fyp*gx fy+fyp]
    Ns = [zeros(n,nx); I; gx]
    Nsx = [zeros(n+nx,nx^2); gxx*kron(hx,Matrix{Float64}(I,nx,nx))]
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
    # p = zeros(nx*ny,nx+ny)
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
    # p = zeros(nx*ny,ny)
    @inbounds for i = 1:ny
        @views p[:,i] = vec(hss[:,:]'*reshape(gxx'[:,i],nx,nx)*hx)
    end
    F5 = Matrix(p')

    F = (-F1 - 2*F2 - F3 -fyp*(F4 + F5))

    F = A\F

    z = dsylvester(B,hx,F)
    hssx = z[1:nx,:]
    gssx = z[nx+1:n,:]



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
        gsss = reshape(z[nx+1:nv,:],ny)

    end

    return (gxxx = gxxx, hxxx = hxxx,
            gssx = gssx, hssx = hssx,
            gsss = gsss, hsss = hsss)
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


function simulation_dsge(model, sol_mat, SS_evaluated, VAR, TS, initial_state, flag_simul, flag_transform)
    @unpack flag_order, flag_deviation, flag_SSestimation, parameters, x, y, xp, yp, variables = model
    @unpack e, eta, f, n, nx, ny, ne, np, SS, skew = model

    Random.seed!(1)

    @unpack gx, hx = sol_mat.fo_mat
    if flag_order > 1
        @unpack gxx, hxx, gss, hss = sol_mat.so_mat
        if flag_order > 2
            @unpack gxxx, hxxx, gssx, hssx, gsss, hsss = sol_mat.to_mat
        end
    end


    if length(initial_state) != nx
        error("The number of inital values for the states must equal the number of states")
    end

    simulated_states_f  =   Array{Float64,2}(undef,nx,TS+1)
    simulated_jumps_f   =   Array{Float64,2}(undef,ny,TS)
    if flag_simul
        simulated_states_f[:,1] = zeros(nx)
    else
        simulated_states_f[:,1] = VAR*ones(nx)
    end
    if flag_order > 1
        simulated_states_s      = Array{Float64,2}(undef,nx,TS+1)
        simulated_jumps_s       = Array{Float64,2}(undef,ny,TS)
        simulated_states_s[:,1] = zeros(nx)
        if flag_order > 2
            simulated_states_t      = Array{Float64,2}(undef,nx,TS+1)
            simulated_jumps_t       = Array{Float64,2}(undef,ny,TS)
            simulated_states_t[:,1] = zeros(nx)
        end
    end

    if flag_simul
        @inbounds for i = 2:TS+1
            #simulated_states_f[:,i]  = sol_mat.fo_mat.hx*simulated_states_f[:,i-1] + (VAR.>0.0)*randn(nx)
            #simulated_states_f[:,i]  = sol_mat.fo_mat.hx*simulated_states_f[:,i-1] + (sqrt.(VAR).>0.0)*randn(nx)
            simulated_states_f[:,i]  = sol_mat.fo_mat.hx*simulated_states_f[:,i-1] + sqrt.(VAR)*randn(nx)
            
            simulated_jumps_f[:,i-1] = sol_mat.fo_mat.gx*simulated_states_f[:,i-1]
            if flag_order > 1
                simulated_states_s[:,i]  = hx*simulated_states_s[:,i-1] + (1/2)*hxx*kron(simulated_states_f[:,i-1],simulated_states_f[:,i-1]) + (1/2)*hss
                simulated_jumps_s[:,i-1] = gx*simulated_states_s[:,i-1] + (1/2)*gxx*kron(simulated_states_f[:,i-1],simulated_states_f[:,i-1]) + (1/2)*gss
                if flag_order > 2
                    simulated_states_t[:,i]  = hx*simulated_states_t[:,i-1] + (1/2)*hxx*kron(simulated_states_f[:,i-1],simulated_states_s[:,i-1]) + (1/6)*hxxx*kron(kron(simulated_states_f[:,i-1],simulated_states_f[:,i-1]),simulated_states_f[:,i-1]) + (3/6)*hssx*simulated_states_f[:,i-1] + (1/6)*hsss
                    simulated_jumps_t[:,i-1] = gx*simulated_states_t[:,i-1] + (1/2)*gxx*kron(simulated_states_f[:,i-1],simulated_states_s[:,i-1]) + (1/6)*gxxx*kron(kron(simulated_states_f[:,i-1],simulated_states_f[:,i-1]),simulated_states_f[:,i-1]) + (3/6)*gssx*simulated_states_f[:,i-1] + (1/6)*gsss
                end
            end
        end
    else
        @inbounds for i = 2:TS+1
            simulated_states_f[:,i]  = hx*simulated_states_f[:,i-1]
            simulated_jumps_f[:,i-1] = gx*simulated_states_f[:,i-1]
            if flag_order > 1
                simulated_states_s[:,i]  = hx*simulated_states_s[:,i-1] + (1/2)*hxx*kron(simulated_states_f[:,i-1],simulated_states_f[:,i-1]) + (1/2)*hss
                simulated_jumps_s[:,i-1] = gx*simulated_states_s[:,i-1] + (1/2)*gxx*kron(simulated_states_f[:,i-1],simulated_states_f[:,i-1]) + (1/2)*gss
                if flag_order > 2
                    simulated_states_t[:,i]  = hx*simulated_states_t[:,i-1] + (1/2)*hxx*kron(simulated_states_f[:,i-1],simulated_states_s[:,i-1]) + (1/6)*hxxx*kron(kron(simulated_states_f[:,i-1],simulated_states_f[:,i-1]),simulated_states_f[:,i-1]) + (3/6)*hssx*simulated_states_f[:,i-1] + (1/6)*hsss
                    simulated_jumps_t[:,i-1] = gx*simulated_states_t[:,i-1] + (1/2)*gxx*kron(simulated_states_f[:,i-1],simulated_states_s[:,i-1]) + (1/6)*gxxx*kron(kron(simulated_states_f[:,i-1],simulated_states_f[:,i-1]),simulated_states_f[:,i-1]) + (3/6)*gssx*simulated_states_f[:,i-1] + (1/6)*gsss
                end
            end
        end
    end

    simulated_states = simulated_states_f
    simulated_jumps  = simulated_jumps_f
    if flag_order > 1
        simulated_states = simulated_states + simulated_states_s
        simulated_jumps  = simulated_jumps + simulated_jumps_s
        if flag_order > 2
            simulated_states = simulated_states + simulated_states_t
            simulated_jumps  = simulated_jumps + simulated_jumps_t
        end
    end

    if flag_transform
        if flag_deviation
            return exp.([simulated_states[:,1:TS]; simulated_jumps[:,1:end]]).*repeat(exp.(SS_evaluated[1:model.n]), 1, TS)
        else
            return SS .+ [simulated_states[:,1:TS]; simulated_jumps[:,1:end]]
        end
    else
        return [simulated_states[:,1:TS]; simulated_jumps[:,1:end]]
    end
end
