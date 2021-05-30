### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 8fefaf75-518f-4684-9cc5-c5e3b3201e02
begin	
	using NLsolve
	using Plots; plotly()#gr()
end

# ╔═╡ 457a4296-bab5-11eb-3675-3148f45e4035
begin
	const bohr = 1.4::Float64
	const w = (9.75*10^3)::Float64
	const h = 1.::Float64
	
	const gzTrue = 2.38::Float64
	const gyTrue = 2.25::Float64
	const gxTrue = 2.25::Float64
	
	const d = -23.89::Float64
	const e = 3.14::Float64
	const lambd = e/d::Float64

	const gzz = gzTrue*(1 + 2/sqrt(1 + 3*lambd^2))::Float64
	const gyy = abs(gyTrue*(1 - (1 + 3*lambd)/sqrt(1 + 3*lambd^2)))::Float64
	const gxx = abs(gxTrue*(1 - (1 - 3*lambd)/sqrt(1 + 3*lambd^2)))::Float64

	const gPAS = [[gxx 0 0]; [0 gyy 0]; [0 0 gzz]]::Matrix{Float64}
	const sig = [[151.5 -8.7 1.5]; [-6.9 131.0 -5.0]; [5.1 -2.8 119.4]]::Matrix{Float64}
	
	
	function pas_lab(alpha::Float64, beta::Float64, gamma::Float64)
		
		rmatrix_hardcoded = ([[cos(alpha)*cos(beta)*cos(gamma) - sin(alpha)*sin(gamma) -cos(alpha)*cos(beta)*sin(gamma) - sin(alpha)*cos(gamma) cos(alpha)*sin(beta)]; [sin(alpha)*cos(beta)*cos(gamma) + cos(alpha)*sin(gamma) -sin(alpha)*cos(beta)*sin(gamma) + cos(alpha)*cos(gamma) sin(alpha)*sin(beta)]; [-sin(beta)*cos(gamma) sin(beta)*sin(gamma) cos(beta)]])::Matrix{Float64}
		
		gLAB = (rmatrix_hardcoded*gPAS*rmatrix_hardcoded')::Matrix{Float64}
		
		Gzz = (gLAB[3,1]^2 + gLAB[3,2]^2 + gLAB[3,3]^2)::Float64
		Gxx = (gLAB[1,1]^2 + gLAB[1,2]^2 + gLAB[1,3]^2)::Float64
		Gxz = (gLAB[1,1]*gLAB[3,1] + gLAB[1,2]*gLAB[3,2] + gLAB[1,3]*gLAB[3,3])::Float64
		
		geff = (sqrt(Gzz))::Float64
		g1 = (sqrt(Gxx - Gxz^2/Gzz))::Float64
		
		return geff, g1
	end
	
	pas_lab(pi/4, pi/6, pi/3)
end

# ╔═╡ 5c475217-ae2a-4e13-9efc-2f4de69d5bb5
begin
	function field_search(F, x, alpha::Float64, field::Float64, gamma::Float64)
		F[1] = h*w/bohr/(pas_lab(alpha, x[1], gamma)[1]) - field
	end
end

# ╔═╡ 2d190990-8c01-4cc1-9c9e-528942512af2
begin
	const start = 992.::Float64
	const stop = 8609.::Float64
	const bin = 50.::Float64
	
	field_array = Array{Float64}(undef, 180)
	Threads.@threads for i=1:180
		a = nlsolve((F,x) -> field_search(F, x, 0.0, start + ((stop-start)/180)*(180-(i)), 0.0), [1.5])
		if converged(a) == true
			field_array[i] = a.zero[1]
		else
			field_array[i] = 0
		end
	end
end

# ╔═╡ b83387e1-f306-43c1-b479-47aeee6b6cee
field_array

# ╔═╡ 4d3df953-608a-4904-88be-9fcc20dfc2e2
plot(field_array)

# ╔═╡ 8e4e1caa-a970-4618-b8a2-66e1d6fcaac3
begin
	const iter = 15.::Float64
	test = [pi - (pi)/i for i=1.:iter]
	#test = [((pi/2)/iter)*(iter-i) for i=1.:iter]
	#test1 = [pi/2 - (pi/2)/i for i=1.:iter]
	#test22 = [((pi/2)/iter)*(iter-i) for i=1.:iter]
	#test = vcat(test1,test22)
	
	test2 = [(2*pi - 2*pi/i) for i=1.:iter]
	test3 = [(2*pi - 2*pi/i) for i=1.:iter]
	#test2 = [(2*pi/iter)*(iter-i) for i=1.:iter]
	#test3 = [(2*pi/iter)*(iter-i) for i=1.:iter]
	#test2 = [(20/10)*(10-v) for v=1:10]
	#field_array
	fie = [(h*w/bohr/pas_lab(j, i, k)[1], (pas_lab(j, i, k)[2])^2) for i=test for j=test2 for k=test3]
	#int = [(pas_lab(j, i, 0.)[2])^2 for i=test for j=test2]
end

# ╔═╡ f8bec11f-fcea-40e7-bbce-03b04e98925e
function gauss(x0::Float64, x, width::Float64)
	return exp(-(x-x0)*(x-x0)/(2*width*width))
end

# ╔═╡ d15171da-b602-4d17-b936-e378ac2fab8c
begin	
	#b = Matrix{Float64}(undef, convert(Int64, (stop - start)÷bin), 2)
	b = zeros(convert(Int64, (stop - start)÷bin), 2)
	Threads.@threads for i=1:convert(Int64, (stop - start)÷bin)
		Threads.@threads for j=1:length(fie)
			if (start + (i-1)*bin) <= fie[j][1] < (start + (i)*bin)
				b[i, 1] = (start + (i-1)*bin)
				b[i, 2] = b[i, 2] + fie[j][2]
			end
		end
	end
end

# ╔═╡ b67cd668-29ae-4f88-9463-a0be1cbf327a
#plot(start:bin:(start+bin*(convert(Int64, (stop - start)÷bin)-1)), b[:,2], xlim=(0,3000), ylim=(0,4500))
plot(start:bin:(start+bin*(convert(Int64, (stop - start)÷bin)-1)), b[:,2])

# ╔═╡ 96b5b0aa-9be3-4048-bdbf-f3c52d6ff7f5
b

# ╔═╡ e340d876-b2d7-4ebe-a9bd-c50209724b2e
#f[x_] := Apply[Plus, MapThread[Times, {Map[Gauss[#, x, 500, 10] &, data6[[All, 1]]], data6[[All, 2]]}]]
function te(x)
	sum(map(y -> gauss(y,x,150.), deleteat!(b[:,1], findall(x->x==0.0, b[:,1]))).*deleteat!(b[:,2], findall(x->x==0.0, b[:,2])))
end

# ╔═╡ 6e3726c7-f6dd-4f2f-b093-8059a2163fc9
plot(te,0,10000)

# ╔═╡ 7b281b23-a958-4762-88d5-17835eaa23e9
begin
	deleteat!(b[:,1], findall(x->x==0.0, b[:,1]))
	deleteat!(b[:,2], findall(x->x==0.0, b[:,2]))
end	


# ╔═╡ Cell order:
# ╠═8fefaf75-518f-4684-9cc5-c5e3b3201e02
# ╠═457a4296-bab5-11eb-3675-3148f45e4035
# ╠═5c475217-ae2a-4e13-9efc-2f4de69d5bb5
# ╠═2d190990-8c01-4cc1-9c9e-528942512af2
# ╠═b83387e1-f306-43c1-b479-47aeee6b6cee
# ╠═4d3df953-608a-4904-88be-9fcc20dfc2e2
# ╠═8e4e1caa-a970-4618-b8a2-66e1d6fcaac3
# ╠═f8bec11f-fcea-40e7-bbce-03b04e98925e
# ╠═d15171da-b602-4d17-b936-e378ac2fab8c
# ╠═b67cd668-29ae-4f88-9463-a0be1cbf327a
# ╠═96b5b0aa-9be3-4048-bdbf-f3c52d6ff7f5
# ╠═e340d876-b2d7-4ebe-a9bd-c50209724b2e
# ╠═6e3726c7-f6dd-4f2f-b093-8059a2163fc9
# ╠═7b281b23-a958-4762-88d5-17835eaa23e9
