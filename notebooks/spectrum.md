---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: julia
    language: julia
    name: julia
---

# Generate theory figures: (Fig. A1 and Fig. A2)
We follow the same approach as the `ParityReadoutSimulator` julia package previouly published as part of the
[azure-quantum-parity-readout github repository](https://github.com/microsoft/azure-quantum-parity-readout)

```python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
```

```python
using OrderedCollections, LinearAlgebra, PyPlot, PyCall
xr = pyimport_conda("xarray", "xarray")
```

```python
using PyCall

# Add the directory to sys.path
py"""
import sys
sys.path.append('/path/to/your/python/file')
"""

# Load common parameters and configs for plotting style
@pyinclude "plotting_helpers.py"
fs = py"font_size"
COLUMNWIDTH = py"COLUMNWIDTH"
```

## Code
### Fermionic basis

```python
# Generic ED code allowing for parity conservation
abstract type AbstractSelector end

struct ParitySelector <: AbstractSelector
    P::Int
end
function (PS::ParitySelector)(v::AbstractVector)
    return (-1)^sum(v) == PS.P
end

struct EverythingSelector <: AbstractSelector end
function (ES::EverythingSelector)(v::AbstractVector)
    return true
end

function buildParitySelector(P::Int)
    if P == 0
        selector = EverythingSelector()
    else
        selector = ParitySelector(P)
    end
    return selector
end

# a chain of fermions
struct FermionBasis
    Nsites::Int

    basis::OrderedDict{Vector{Int},Int}

    function FermionBasis(Nsites, basis_selector)
        full_basis = Iterators.product(fill(0:1, Nsites)...)

        basis = OrderedDict{Vector{Int},Int}()
        for basis_vec_ in full_basis
            basis_vec = collect(basis_vec_)
            if basis_selector(basis_vec)
                basis[basis_vec] = length(basis) + 1
            end
        end

        return new(Nsites, basis)
    end
end

Base.length(basis::FermionBasis) = length(basis.basis)
Base.size(basis::FermionBasis) = length(basis.basis)

function fermion_site_parity_op(basis::FermionBasis, m::Int)
    @assert 1 <= m <= basis.Nsites
    return diagm(0 => [(-1)^v[m] for (v,_) in basis.basis])
end

function fermion_site_number_op(basis::FermionBasis, m::Int)
    @assert 1 <= m <= basis.Nsites
    return diagm(0 => [v[m] for (v,_) in basis.basis])
end

function fermion_hop(basis::FermionBasis, to::Int, from::Int)
    @assert 1 <= from <= basis.Nsites
    @assert 1 <= to <= basis.Nsites
    @assert to != from

    R = zeros(length(basis), length(basis))
    for (state_vec, state_idx) in basis.basis
        if state_vec[from] == 1 && state_vec[to] == 0
            new_state_vec = copy(state_vec)
            new_state_vec[from] = 0
            new_state_vec[to]   = 1

            new_state_idx = get(basis.basis, new_state_vec, nothing)

            parity_sites = min(from,to):max(from,to)
            parity = -(-1)^sum(state_vec[parity_sites])

            if new_state_idx !== nothing
                R[new_state_idx, state_idx] = parity
            end
        end
    end

    return R
end

function fermion_pair_create(basis::FermionBasis, pos1::Int, pos2::Int)
    @assert 1 <= pos1 <= basis.Nsites
    @assert 1 <= pos2 <= basis.Nsites
    @assert pos1 != pos2

    if pos2 < pos1
        return -fermion_pair_create(basis, pos2, pos1)
    end

    R = zeros(length(basis), length(basis))
    for (state_vec, state_idx) in basis.basis
        if state_vec[pos1] == state_vec[pos2] == 0
            new_state_vec = copy(state_vec)
            new_state_vec[pos1] = 1
            new_state_vec[pos2] = 1

            new_state_idx = get(basis.basis, new_state_vec, nothing)

            parity_sites = min(pos1,pos2)+1:max(pos1,pos2)-1
            parity = (-1)^sum(state_vec[parity_sites])

            if new_state_idx !== nothing
                R[new_state_idx, state_idx] = parity
            end
        end
    end

    return R
end

function fermion_pair_annihilate(basis::FermionBasis, pos1::Int, pos2::Int)
    return -fermion_pair_create(basis, pos2, pos1)'
end

function majorana_hopping(basis::FermionBasis, pos1::Int, pos2::Int)
    if pos1 == pos2
        return im*I(size(basis))
    end

    i = div(pos1+1, 2)
    j = div(pos2+1, 2)

    if i == j
        return (pos1 < pos2 ? +1 : -1) * fermion_site_parity_op(basis, i)
    end

    if pos1 % 2 == 1 && pos2 % 2 == 1
        O = im*(fermion_hop(basis, i, j) + fermion_pair_create(basis, i, j))
    elseif pos1 % 2 == 1 && pos2 % 2 == 0
        O = -fermion_hop(basis, i, j) + fermion_pair_create(basis, i, j)
    elseif pos1 % 2 == 0 && pos2 % 2 == 1
        O = fermion_hop(basis, i, j) + fermion_pair_create(basis, i, j)
    else
        O = im*(fermion_hop(basis, i, j) - fermion_pair_create(basis, i, j))
    end
    return O + O'
end

function majorana_dot_coupling_op(basis::FermionBasis, majorana_index, dot_index, Nmaj)
    @assert 0 < majorana_index <= Nmaj
    @assert Nmaj %2 ==0 # Total number of Majoranas in basis should be even
    @assert dot_index+div(Nmaj,2) <= basis.Nsites

    m_fermion_index = div(majorana_index+1, 2)
    d_fermion_index = div(Nmaj,2)+dot_index
    if majorana_index % 2 == 1
        op = im*(
            fermion_hop(basis, m_fermion_index, d_fermion_index) -
            fermion_pair_annihilate(basis, m_fermion_index, d_fermion_index)
        )
    else
        op = (
            fermion_hop(basis, m_fermion_index, d_fermion_index) +
            fermion_pair_annihilate(basis, m_fermion_index, d_fermion_index)
        )
    end
    return op
end
```

### Model

```python
"""
Basis convention :
* All majoranas are first then the QD level.
"""
struct XMPR_Hamiltonian
    H::Matrix{ComplexF64}
    EC::Float64
    Dot_number_op::Matrix{ComplexF64}

    function XMPR_Hamiltonian(;
        EC::Float64, parity::Int,
        tm1::Float64, tm2::Float64, tm2_phase::Float64, ng::Float64, e12::Float64, e34::Float64
    )
        Nmaj = 4
        Nfermions = 3
        selector = buildParitySelector(parity)
        fermion_basis = FermionBasis(Nfermions, selector)
        dot_index = 3
        dot_number_op = fermion_site_number_op(fermion_basis, dot_index)

        H0 = zeros(ComplexF64, length(fermion_basis), length(fermion_basis))
        maj_indices= [1,3,2,4] #

        H0 += e12*majorana_hopping(fermion_basis, maj_indices[1], maj_indices[2])
        H0 += e34*majorana_hopping(fermion_basis, maj_indices[3], maj_indices[4])

        op_tm1 = tm1 * majorana_dot_coupling_op(fermion_basis, maj_indices[1], 1, Nmaj)
        H0 += op_tm1 + op_tm1'
        op_tm2 = tm2 * exp(im*π*tm2_phase) * majorana_dot_coupling_op(fermion_basis, maj_indices[3], 1, Nmaj)
        H0 += op_tm2 + op_tm2'

        for k in 1:size(H0, 1)
            H0[k,k] += EC*(dot_number_op[k,k] - ng)^2
        end

        return new(H0, EC, dot_number_op)
    end
end

function calculate_CQ_prefactor(Ec, lever_arm)
    units = 80.1088317 # e^2 / (2 ueV) in fF
    return lever_arm^2*units/Ec
end
```

### Sweep functions

```python
function evaluate_spectrum(params)
    return eigvals(Hermitian(XMPR_Hamiltonian(;params...).H))
end

function evaluate_dot_occupation(params)
    model = XMPR_Hamiltonian(;params...)
    op = model.Dot_number_op
    vals, vecs = eigen(Hermitian(model.H))
    return [real(dot(vecs[:,i],op,vecs[:,i])) for i in 1:size(vecs,2)]
end

function generate_data_grid(sweep_params, fixed_params, f)
    # Check that there's no overlap between swept and fixed parameters
    @assert length(intersect(keys(sweep_params), keys(fixed_params)))==0
    params = OrderedDict([k=>v[1] for (k,v) in sweep_params]..., [k=>v for (k,v) in fixed_params]...)
    test_results = f(params)
    res_dims = size(test_results)
    dims = length.(values(sweep_params))
    Ndims = length(sweep_params)

    # Use one OrderedSDict with the first Ndims elements the swept parameters
    data = zeros(Float64, prod(dims), res_dims...)
    for (j,pt) in enumerate(Iterators.product(values(sweep_params)...))
        for (n,k) in enumerate(keys(sweep_params))
            params[k] = pt[n]
        end
        data[j,:] = f(params)
    end
    return xr.DataArray(reshape(data, dims..., res_dims...), dims=vcat(collect(keys(sweep_params)), "index"), coords=sweep_params);
end
```

## Figures and data generation


### Figure A1 : Spectrum vs Ed

```python
# Generating the data for figure A1:
fixed_params = Dict(:EC=>150.0,:tm2=>2.5,:tm1=>2.0)
sweep_params = Dict(:ng=>0.4:0.001:0.6, :e12=>[0.0,0.5], :e34=>[0.0,0.5], :tm2_phase=>[0.0, 0.5],:parity=>[1,-1])
spec = generate_data_grid(sweep_params, fixed_params, evaluate_spectrum);
```

```python
lstyle=Dict(
    1=>Dict(:ls=>"-", :lw=>1),
    -1=>Dict(:ls=>"--",:lw=>1)
)

fig, axs = plt.subplots(2,2, figsize=(COLUMNWIDTH,2.8), squeeze=false, sharey=:row, sharex=true,layout="constrained")

# Convert from ng to detuning energy and mask to plot only near resonance
Ed_bnd = 11.0 # Bnds on Ed for plotting
x =  fixed_params[:EC]*(1 .- 2*spec.ng.values)
m = abs.(x) .<=Ed_bnd
x= x[m]

for (j, eM) in enumerate([0.5, 0.0]), (jphi, phi) in enumerate([0.5, 0.0])
    ax=axs[jphi,j]
    for p in [1,-1]
        ax.set_xlim(-10,10)
        y = spec.sel(parity=p, e12=eM, e34=eM, tm2_phase=phi).values[m,:]
        ax.plot(x,y[:,1], color="tab:red", label=L"+1"; lstyle[p]...)
        ax.plot(x,y[:,4], color="tab:red"; lstyle[p]...)

        ax.plot(x,y[:,2], color="tab:blue", label=L"-1", ;lstyle[p]...)
        ax.plot(x,y[:,3], color="tab:blue", ;lstyle[p]...)

        # Arrows
        # We only plot the arrows for the top row and once per panel.
        (p==-1 || jphi==2) ? continue : nothing
        y2 = maximum(y[:,2])
        y1 = maximum(y[:,1])

        head_size=0.75
        ax.arrow(x=-0.5, y=y2, dx=0.0, dy=y1-y2+head_size, width=0.03,
                 head_width=head_size, head_length=head_size, fc="black", ec="black", lw=1.5,zorder=10)

        ax.arrow(x=+0.5, y=y1, dx=0.0, dy=y2-y1-head_size, width=0.03,
                 head_width=head_size, head_length=head_size, fc="gray", ec="gray", lw=1.5, zorder=10)
    end
end

# Titles and labels
axs[1,1].set_title(L"(a) $E_{12}=E_{34} =0.5$ ${\rm \mu}$eV", fontsize=fs, loc="left")
axs[1,2].set_title(L"(b) $E_{12}=E_{34} =0$ ${\rm \mu}$eV", fontsize=fs, loc="left")
axs[2,1].set_title("(c)", fontsize=fs, loc="left")
axs[2,2].set_title("(d)", fontsize=fs, loc="left")

axs[2,1].set_ylabel(L"Spectrum [${\rm \mu}$eV]")
axs[1,1].set_ylabel(L"Spectrum [${\rm \mu}$eV]")
axs[2,1].set_xlabel(L"$E_\mathrm{D}$ [${\rm \mu}$eV]")
axs[2,2].set_xlabel(L"$E_\mathrm{D}$ [${\rm \mu}$eV]")

# Add row labels
a2 = axs[1,2].twinx()
a2.set_yticks([])
a2.set_ylabel(L"\varphi=\pi/2")
a2_0p5 = a2

a2 = axs[2,2].twinx()
a2.set_yticks([])
a2.set_ylabel(L"\varphi=\pi")
a2_0 = a2

# Make the frame and tick colors match the line cuts of the other plot.
colors = ["tab:green", "tab:purple"]
for (j, c) in enumerate(colors)
    for ax in axs[j,:]
        ax.tick_params(color=c, which="both")
        for spine in ax.spines.values()
            spine.set_edgecolor(c)
        end
    end
end
for (axes, color) in zip([a2_0p5, a2_0], colors)
    for spine in axes.spines.values()
        spine.set_edgecolor(color)
    end
end

plt.savefig("figures/spectrum_detuning.pdf", bbox_inches="tight", pad_inches=0.01, transparent=true)
```

### Figure A2: Spectrum vs phi

```python
# Generating the data for figure A2:
fixed_params = Dict(:EC=>150.0,:tm2=>2.5,:tm1=>2.0)
sweep_params = Dict(:ng=>0.47:0.001:0.51,:e12=>[0.0,0.5], :e34=>[0.0,0.5],  :tm2_phase=>collect(0.0:0.01:2.0), :parity=>[1,-1])
@time spec_phi = generate_data_grid(sweep_params, fixed_params, evaluate_spectrum);
@time occ_phi = generate_data_grid(sweep_params, fixed_params, evaluate_dot_occupation);
```

```python
fig, axs = plt.subplots(2,1,figsize=(COLUMNWIDTH,2.8), sharex=true, layout="constrained")

#######################
# Panel (a): spectrum #
#######################
ng0 = 0.5
ax=axs[1]
parity = 1

# Finite Em
x = spec_phi.tm2_phase.values
y = spec_phi.sel(parity=parity, e12=0.5, e34=0.5, ng=ng0).isel(index=0:3).values
ax.plot(x,y[:,1],c="tab:orange", label=L"0.5")
for j in 2:4
    ax.plot(x,y[:,j],c="tab:orange")
end

# Em=0
y = spec_phi.sel(parity=parity, e12=0.0, e34=0.0, ng=ng0).isel(index=0:3).values
ax.plot(x,y[:,1],ls=":", c="k", label=L"0.0")
for j in 2:4
    ax.plot(x,y[:,j],ls=":", c="k")
end

# Titles, labels
lg = ax.legend(title=L"$E_{12}=E_{34}$ [${\rm \mu}$eV]",ncol=2, bbox_to_anchor=[0.5, 1.0], loc="lower left", fontsize=fs)
plt.setp(lg.get_title(),fontsize=fs)
ax.set_ylabel(L"Spectrum [${\rm \mu}$eV]")
ax.set_title(L"(a) $E_\mathrm{D} =%$(round(200*(1-2*ng0); digits=2))$ ${\rm \mu}$eV", fontsize=fs, loc="left")

##################
#  Panel (b): CQ #
##################
ng0 = 0.48
Ed = fixed_params[:EC]*(1-2*ng0)
@show Ed
ax=axs[2]

prefactor = calculate_CQ_prefactor(fixed_params[:EC], 0.3)

# Finite Em
data = prefactor*occ_phi.differentiate(:ng).sel(ng=ng0, e12=0.5, e34=0.5, parity=parity).isel(index=0:1).squeeze(drop=true)
data.plot.line(x=:tm2_phase, ax=ax, add_legend=false, c="tab:orange")

# Em = 0
data = prefactor*occ_phi.differentiate(:ng).sel(ng=ng0, e12=0.0, e34=0.0, parity=parity).isel(index=0:1).squeeze(drop=true)
data.plot.line(x=:tm2_phase, ax=ax, c="k", ls=":", add_legend=false)

# Titles, labels
ax.set_title(L"(b) $E_\mathrm{D} =%$(round(Ed; digits=1))$ ${\rm \mu}$eV", fontsize=fs, loc="left")
ax.set_title("")
ax.set_xlabel(L"\varphi/\pi")
ax.set_ylabel(L"$C_{\rm Q}$ [fF]")
axs[2].set_ylim(0,0.6)

# Line cuts for other panel
for a in axs
    a.axvline(0.5, c="tab:green",ls="--")
    a.axvline(1.0, c="tab:purple", ls="--")
    a.set_xlim(0,2)
end

plt.savefig("figures/spectrum_flux.pdf", bbox_inches="tight", pad_inches=0.01, transparent=true)
```

```python

```
