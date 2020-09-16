using Gridap
using LinearAlgebra: inv, det
using LineSearches: BackTracking, StrongWolfe, HagerZhang
using BenchmarkTools

E = 10.
ν = 0.3
const μ = E/2/(1+ν)
const λ = 2*μ*ν/(1-2*ν)

function run()

    # deformation gradient

    F(∇u) = one(∇u) + ∇u
    J(F) = det(F)
    C(F) = (F') * F

    # body force
    B(x) = VectorValue(
        0.0, -0.5, 0.0
    )

    # surfaceTraction
    T(x) = VectorValue(
        -0.1, 0.0, 0.0
    )

    # constitutive relation
    @law function S(∇u)
        Js = J(F(∇u))
        μ * (F(∇u) - (inv(F(∇u)))') + λ * Js * (Js - 1) * (inv(F(∇u)))' 
    end

    function res(u, v)
        S(∇(u)) ⊙ ∇(v) - B ⊙ v
    end

    @law function Lv(∇du, ∇u)
        Js = J(F(∇u))
        Finv = inv(F(∇u))
        μ * (∇du + Finv' ⋅ ∇du' ⋅ Finv') + λ * (2. * Js - 1.) * Js * ((Finv' ⊙ ∇du) * Finv') - λ * Js * (Js - 1) * (Finv' ⋅ ∇du' ⋅ Finv') 
    end

    jac(u, du, v) = Lv(∇(du), ∇(u)) ⊙ ∇(v)

    domain = (0, 1, 0, 1, 0, 1)
    partition = (24, 16, 16)
    tmodel = CartesianDiscreteModel(domain, partition)
    model = simplexify(tmodel)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"rightFace",[2, 4, 6, 8, 14, 16, 18, 20, 26])
    add_tag_from_tags!(labels,"leftFace",[1, 3, 5, 7, 13, 15, 17, 19, 25])

    V = TestFESpace(
    model=model,valuetype=VectorValue{3,Float64},
    reffe=:Lagrangian,conformity=:H1,order=1,
    dirichlet_tags = ["leftFace", "rightFace"])

    trian = Triangulation(model)
    degree=2
    quad=CellQuadrature(trian, degree)

    neumannTags = ["boundary"]
    btrian = BoundaryTriangulation(model, neumannTags)
    bquad = CellQuadrature(btrian, degree)

    a_Ω = FETerm(res, jac, trian, quad)

    b_Γ(v) =  T ⊙ v
    t_Γ = FESource(b_Γ, btrian, bquad)

    nls = NLSolver(
    show_trace=true,
    method=:newton,
    linesearch=BackTracking()
    )

    g0(x) = VectorValue(0.0,0.0,0.0)
    g1(x) = VectorValue(
        0.0,
        (0.5 + (x[2]-0.5) * cos(π/3) - (x[3] - 0.5) * sin(π/3) - x[2])/2,
        (0.5 + (x[2]-0.5) * sin(π/3) + (x[3] - 0.5) * cos(π/3) - x[3])/2
    )

    U = TrialFESpace(V, [g0, g1])

    #FE problem
    op = FEOperator(U, V, a_Ω, t_Γ)
    solver = FESolver(nls)

    x0 = zeros(Float64, num_free_dofs(V))
    uh = FEFunction(U, x0)
    uh, = solve!(uh, solver, op)

    writevtk(trian,"results_hyper",cellfields=["uh"=>uh,"sigma"=>S(∇(uh))])
end
@btime run()