### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ ff595197-fb8f-411b-a63d-33d1b5a683d8
begin
	using LinearAlgebra
	using DifferentialEquations
	using BenchmarkTools
	using Plots; gr()
end

# ╔═╡ 002131c0-b897-11eb-3c23-311bcd7e97a6
begin
	
	# General
	const bohr = 1.4::Float64
	const h = 1.::Float64
	const w = (9.75*10^3)::Float64
	const kbolt = (2.08366*10^4)::Float64
	const gzTrue = 2.38::Float64
	const gyTrue = 2.25::Float64
	const gxTrue = 2.25::Float64
	const d = -23.89::Float64
	const e = 3.14::Float64
	lambd = e/d

	#const B0 = 1945.4176999856793::Float64
	#const B0 = 991.8868351942978::Float64
	const B1 = 0.001878::Float64
	
	# g-factor
	gzz = abs(gzTrue*(1+2/sqrt(1+3*lambd^2)))
	gyy = abs(gyTrue*(1-(1+3*lambd)/sqrt(1+3*lambd^2)))
	gxx = abs(gxTrue*(1-(1-3*lambd)/sqrt(1+3*lambd^2)))
	@show [gzz, gyy, gxx]
	
	# Orientation
	#const alpha = 0.::Float64
	#const beta = (pi/4)::Float64
	#const gamma = 0.::Float64
	#@show [alpha, beta, gamma]
	
	# Time and Delay
	const delay = 15000.::Float64
	const pulse = 400.::Float64
	const Tcryo = 10.::Float64
	const Tterm = 7000.::Float64
	const heatcapacity = 10.::Float64
	const heatparameter = 12.105::Float64
	
	# Relaxation
	const T1 = 750.::Float64
	const T2 = 1.147::Float64
	@show [T1, T2]
	
	# Spin matrices
	sx = [0. 1/2 + 0im; 1/2 + 0im 0.]
	sy = [0. (0. - 1im)/2; (0. + 1im)/2 0.]
	sz = [1/2 + 0im 0.; 0. -1/2 + 0im]
	idmat = [1 + 0im 0.; 0. 1 + 0im]
	
	# Hamiltonian
	k = [0. 0. 1.]
	k1 = [1. 0. 0.]
	svector = [sx, sy, sz]
	
	#Relaxation
	relaxMatrix = [[-1/(2*T1) 0. 0. + 0im 1/(2*T1)]; [0. -1/(T2) 0. 0.]; [0. 0. -1/(T2) 0.]; [1/(2*T1) 0. 0. -1/(2*T1)]]
	
	# Temperature
	Tafterpulse = Tcryo + (heatparameter/pulse)*pulse/heatcapacity
	
	
	
	# PAS to LAB
	gPAS = [[gxx 0. 0.]; [0. gyy 0.]; [0. 0. gzz]]
	
	function glab_test(alpha, beta, gamma)
	
	function hamiltonian(alpha, beta, gamma)
		
		rmatrix_hardcoded = [[cos(alpha)*cos(beta)*cos(gamma) - sin(alpha)*sin(gamma) -cos(alpha)*cos(beta)*sin(gamma) - sin(alpha)*cos(gamma) cos(alpha)*sin(beta)]; [sin(alpha)*cos(beta)*cos(gamma) + cos(alpha)*sin(gamma) -sin(alpha)*cos(beta)*sin(gamma) + cos(alpha)*cos(gamma) sin(alpha)*sin(beta)]; [-sin(beta)*cos(gamma) sin(beta)*sin(gamma) cos(beta)]]
		gLAB = (rmatrix_hardcoded')*gPAS*rmatrix_hardcoded
		
		Gzz = gLAB[3,1]^2 + gLAB[3,2]^2 + gLAB[3,3]^2
		Gxx = gLAB[1,1]^2 + gLAB[1,2]^2 + gLAB[1,3]^2
		Gxz = gLAB[1,1]*gLAB[3,1] + gLAB[1,2]*gLAB[3,2] + gLAB[1,3]*gLAB[3,3]
		geff = sqrt(Gzz)
		g1 = sqrt(Gxx - Gxz^2/Gzz)
		
		#currently unused
		klab = [gLAB[3,1] gLAB[3,2] gLAB[3,3]]/geff
		k1lab = [gLAB[1,1] - (gLAB[3,1]*(gLAB[1,1]*gLAB[3,1] + gLAB[1,2]*gLAB[3,2] + gLAB[1,3]*gLAB[3,3]))/geff^2 gLAB[1,2] - (gLAB[3,2]*(gLAB[1,1]*gLAB[3,1] + gLAB[1,2]*gLAB[3,2] + gLAB[1,3]*gLAB[3,3]))/geff^2 gLAB[1,3] - (gLAB[3,3]*(gLAB[1,1]*gLAB[3,1] + gLAB[1,2]*gLAB[3,2] + gLAB[1,3]*gLAB[3,3]))/geff^2]/g1
		
		B0 = h*w/(geff*bohr)
		hamiltonian_calc = -(bohr*B0*geff - h*w)*k*svector + bohr*B1*g1*k1*svector
	
		h0 = -bohr*B0*geff*k*svector
		rHamiltonian0 = kron(idmat, conj(h0[1])) #conjugate transpose?! # there is something strange in the definition of h0 and other matrices that calculated using mu;tiplication of a vector and matrix
	
		liouvillian = -kron(hamiltonian_calc[1], idmat) + kron(idmat, conj(hamiltonian_calc[1])) #conjugate transpose?!
		
		return liouvillian, rHamiltonian0, h0, gLAB, klab
	end
	
	#@show h*w/(geffective(alpha, beta, gamma)[1]*bohr)
	
end

# ╔═╡ 71024f7a-2319-430f-ba43-9f561e5ce68a
begin
	function temperature(x)
		if 0 <= x < delay
			return Tcryo
		elseif delay <= x < delay + pulse
			return Tcryo + (heatparameter/pulse)*(x - delay)/heatcapacity
		elseif x > delay + pulse
			return (Tafterpulse - Tcryo)*exp(-(x - delay - pulse)/Tterm) + Tcryo
		else
			return Tcryo
		end
	end
	
	function relaxMatrixEq(x, alpha, beta, gamma)
		if 0 <= x < delay
			return relaxMatrix*exp(hamiltonian(alpha, beta, gamma)[2]/(kbolt*Tcryo))
		elseif delay <= x < delay + pulse
			return relaxMatrix*exp(hamiltonian(alpha, beta, gamma)[2]/(kbolt*(Tcryo + (heatparameter/pulse)*(x - delay)/heatcapacity)))
		elseif x > delay + pulse
			return relaxMatrix*exp(hamiltonian(alpha, beta, gamma)[2]/(kbolt*((Tafterpulse - Tcryo)*exp(-(x - delay - pulse)/Tterm) + Tcryo)))
		else
			return relaxMatrix*exp(hamiltonian(alpha, beta, gamma)[2]/(kbolt*Tcryo))
		end
	end
	
	function roEqt(x, alpha, beta, gamma)
		temp = exp(-(hamiltonian(alpha, beta, gamma)[3])[1]/(kbolt*temperature(x)))
		return temp/tr(temp)
	end
end

# ╔═╡ fd9c23ab-08ea-4fd4-a855-3fda53fb70ca
function model_kuprov(du, u, par, t)
	alpha, beta, gamma = par
	
	temp = (-1im*hamiltonian(alpha, beta, gamma)[1] + relaxMatrixEq(t, alpha, beta, gamma)).*([u[1] u[2] u[3] u[4]])
    du[1] = sum(temp[1,:])
    du[2] = sum(temp[2,:])
    du[3] = sum(temp[3,:])
	du[4] = sum(temp[4,:])
end

# ╔═╡ b61924f6-5af0-4102-ba20-0afa0ac69545
function model_standard(du, u, par, t)
	alpha, beta, gamma = par
	
	temp1 = (-1im*hamiltonian(alpha, beta, gamma)[1]).*([u[1] u[2] u[3] u[4]])
	temp2 = (relaxMatrix).*([u[1] u[2] u[3] u[4]]) - (relaxMatrix).*([roEqt(t, alpha, beta, gamma)[1] roEqt(t, alpha, beta, gamma)[2] roEqt(t, alpha, beta, gamma)[3] roEqt(t, alpha, beta, gamma)[4]])
	
    du[1] = sum(temp1[1,:]) + sum(temp2[1,:])
    du[2] = sum(temp1[2,:]) + sum(temp2[2,:])
    du[3] = sum(temp1[3,:]) + sum(temp2[3,:])
	du[4] = sum(temp1[4,:]) + sum(temp2[4,:])
end

# ╔═╡ 485220ef-2ff8-4ff9-a60f-27f399199961
begin
	par = (pi/3, pi/3, 0) # we could also make this an array, or any other sequence type!
	u0 = [roEqt(0, par[1], par[2], par[3])[1] roEqt(0, par[1], par[2], par[3])[2] roEqt(0, par[1], par[2], par[3])[3] roEqt(0, par[1], par[2], par[3])[4]]
	tspan = (0., 100000.0)
	#prob = ODEProblem(model_kuprov, u0, tspan)
	prob = ODEProblem(model_kuprov, u0, tspan, par)
	#sol = solve(prob, Rosenbrock32(autodiff=false),abstol=1e-9,reltol=1e-9, saveat=3)
	#https://github.com/SciML/DifferentialEquations.jl/issues/110
	@benchmark solve(prob, Rosenbrock32(autodiff=false),abstol=1e-9,reltol=1e-9, saveat=3)
end

# ╔═╡ 24727341-166f-421d-8d7e-9d7a805bf52d
begin
	plot(sol)
end

# ╔═╡ e68720b1-532c-4dd0-a818-4d88c7634ca7
begin
	# klab is (hamiltonian(par[1], par[2], par[3])[5])
	# gLAB is (hamiltonian(par[1], par[2], par[3])[4])
	
	# Absorption
	mxlabsin = real(0.25*2*(1im*(hamiltonian(par[1], par[2], par[3])[4])[1,3]*(hamiltonian(par[1], par[2], par[3])[5])[1]*(sol[2,:] - sol[3,:]) - 1im*(hamiltonian(par[1], par[2], par[3])[4])[1,1]*(hamiltonian(par[1], par[2], par[3])[5])[3]*(sol[2,:] - sol[3,:]) - (hamiltonian(par[1], par[2], par[3])[4])[1,3]*(hamiltonian(par[1], par[2], par[3])[5])[2]*(sol[2,:] + sol[3,:]) + (hamiltonian(par[1], par[2], par[3])[4])[1,2]*(hamiltonian(par[1], par[2], par[3])[5])[3]*(sol[2,:] + sol[3,:]) + (hamiltonian(par[1], par[2], par[3])[4])[1,1]*(hamiltonian(par[1], par[2], par[3])[5])[2]*(sol[1,:] - sol[4,:]) + (hamiltonian(par[1], par[2], par[3])[4])[1,2]*(hamiltonian(par[1], par[2], par[3])[5])[1]*(-sol[1,:] + sol[4,:])))

	# Dispertion
	mxlabcos = real(0.25*(1im*(hamiltonian(par[1], par[2], par[3])[4])[1,2]*((1 + (hamiltonian(par[1], par[2], par[3])[5])[1]^2 + (hamiltonian(par[1], par[2], par[3])[5])[3]^2)*(sol[2,:] - sol[3,:]) + (hamiltonian(par[1], par[2], par[3])[5])[2]^2*(-sol[2,:] + sol[3,:]) + 2im*(hamiltonian(par[1], par[2], par[3])[5])[2]*((hamiltonian(par[1], par[2], par[3])[5])[3]*sol[1,:] + (hamiltonian(par[1], par[2], par[3])[5])[1]*(sol[2,:] + sol[3,:]))) + 2*(hamiltonian(par[1], par[2], par[3])[4])[1,2]*(hamiltonian(par[1], par[2], par[3])[5])[2]*(hamiltonian(par[1], par[2], par[3])[5])[3]*sol[4,:] + (hamiltonian(par[1], par[2], par[3])[4])[1,3]*((1 + (hamiltonian(par[1], par[2], par[3])[5])[1]^2 + (hamiltonian(par[1], par[2], par[3])[5])[2]^2 - (hamiltonian(par[1], par[2], par[3])[5])[3]^2)*sol[1,:] - 2*(hamiltonian(par[1], par[2], par[3])[5])[3]*(1im*(hamiltonian(par[1], par[2], par[3])[5])[2]*(sol[2,:] - sol[3,:]) + (hamiltonian(par[1], par[2], par[3])[5])[1]*(sol[2,:] + sol[3,:])) - (1 + (hamiltonian(par[1], par[2], par[3])[5])[1]^2 + (hamiltonian(par[1], par[2], par[3])[5])[2]^2 - (hamiltonian(par[1], par[2], par[3])[5])[3]^2)*sol[4,:]) + (hamiltonian(par[1], par[2], par[3])[4])[1,1]*(-2im*(hamiltonian(par[1], par[2], par[3])[5])[1]*(hamiltonian(par[1], par[2], par[3])[5])[2]*(sol[2,:] - sol[3,:]) - (hamiltonian(par[1], par[2], par[3])[5])[1]^2*(sol[2,:] + sol[3,:]) + (1 + (hamiltonian(par[1], par[2], par[3])[5])[2]^2 + (hamiltonian(par[1], par[2], par[3])[5])[3]^2)*(sol[2,:] + sol[3,:]) + 2*(hamiltonian(par[1], par[2], par[3])[5])[1]*(hamiltonian(par[1], par[2], par[3])[5])[3]*(-sol[1,:] + sol[4,:]))))
	
	# free term
	mxlab = real(0.25*(-1im*(hamiltonian(par[1], par[2], par[3])[4])[1,2]*((-1 + (hamiltonian(par[1], par[2], par[3])[5])[1]^2 + (hamiltonian(par[1], par[2], par[3])[5])[3]^2)*(sol[2,:] - sol[3,:]) +  (hamiltonian(par[1], par[2], par[3])[5])[2]^2*(-sol[2,:] + sol[3,:]) + 2im*(hamiltonian(par[1], par[2], par[3])[5])[2]*((hamiltonian(par[1], par[2], par[3])[5])[1]*(sol[2,:] + sol[3,:]) + (hamiltonian(par[1], par[2], par[3])[5])[3]*(sol[1,:] - sol[4,:]))) + (hamiltonian(par[1], par[2], par[3])[4])[1,1]*(2im*(hamiltonian(par[1], par[2], par[3])[5])[1]*(hamiltonian(par[1], par[2], par[3])[5])[2]*(sol[2,:] - sol[3,:]) + (hamiltonian(par[1], par[2], par[3])[5])[1]^2*(sol[2,:] + sol[3,:]) - (-1 + (hamiltonian(par[1], par[2], par[3])[5])[2]^2 + (hamiltonian(par[1], par[2], par[3])[5])[3]^2)*(sol[2,:] + sol[3,:]) + 2*(hamiltonian(par[1], par[2], par[3])[5])[1]*(hamiltonian(par[1], par[2], par[3])[5])[3]*(sol[1,:] - sol[4,:])) + (hamiltonian(par[1], par[2], par[3])[4])[1,3]*(-(-1 + (hamiltonian(par[1], par[2], par[3])[5])[1]^2 + (hamiltonian(par[1], par[2], par[3])[5])[2]^2 -  (hamiltonian(par[1], par[2], par[3])[5])[3]^2)*sol[1,:] + 2im*(hamiltonian(par[1], par[2], par[3])[5])[2]*(hamiltonian(par[1], par[2], par[3])[5])[3]*(sol[2,:] - sol[3,:]) + 2*(hamiltonian(par[1], par[2], par[3])[5])[1]*(hamiltonian(par[1], par[2], par[3])[5])[3]*(sol[2,:] + sol[3,:]) + (-1 + (hamiltonian(par[1], par[2], par[3])[5])[1]^2 + (hamiltonian(par[1], par[2], par[3])[5])[2]^2 - (hamiltonian(par[1], par[2], par[3])[5])[3]^2)*sol[4,:])))
	
end

# ╔═╡ 5854d54c-41a2-4e7f-9c5b-7ff9c982b7d1
#plot(mxlabsin, ylims = (-9*10^-5, -7.5*10^-5), xlims = (14000/3, 18000/3))
plot(mxlabsin)
#, xlims = (14000/3, 18000/3)

# ╔═╡ Cell order:
# ╠═ff595197-fb8f-411b-a63d-33d1b5a683d8
# ╠═002131c0-b897-11eb-3c23-311bcd7e97a6
# ╟─71024f7a-2319-430f-ba43-9f561e5ce68a
# ╟─fd9c23ab-08ea-4fd4-a855-3fda53fb70ca
# ╟─b61924f6-5af0-4102-ba20-0afa0ac69545
# ╠═485220ef-2ff8-4ff9-a60f-27f399199961
# ╠═24727341-166f-421d-8d7e-9d7a805bf52d
# ╟─e68720b1-532c-4dd0-a818-4d88c7634ca7
# ╠═5854d54c-41a2-4e7f-9c5b-7ff9c982b7d1
