# export JULIA_NUM_THREADS=2  # Set this in the terminal before running the script

println(Sys.CPU_THREADS)
println("Number of threads = ", Threads.nthreads())


using CSV, DataFrames, Random, Plots
using SymbolicRegression
using Serialization

# println(pwd())
# println("Script directory: ", @__DIR__)

cd(@__DIR__)
df = CSV.read("CSV_OLD/interfacial_results_cleaned.csv", DataFrame)

target  = ["P_bubble", "P_dew"]
exclude = vcat(
    ["gamma", "interfacial_thickness", "E_carbon dioxide", "E_argon", "E_hydrogen"],
    ["x_carbon dioxide", "x_argon", "x_hydrogen", "y_carbon dioxide", "y_argon", "y_hydrogen"],
    ["z_carbon dioxide"],
    ["liquid_density", "vapor_density"],
    ["pressure"],
    target,
)

features = [col for col in names(df) if col ∉ exclude]
println("Features: ", features)

X = Matrix{Float64}(df[:, features])          # (n_samples, n_features)
y_bubble = Vector{Float64}(df[:, "P_bubble"])
y_dew    = Vector{Float64}(df[:, "P_dew"])

# ── Train / test split (80/20, seed=85) ──────────────────────────────────────
n = size(X, 1)
idx = randperm(MersenneTwister(85), n)
n_train = round(Int, 0.8 * n)
train_idx = idx[1:n_train]
test_idx  = idx[n_train+1:end]

X_train = X[train_idx, :]
X_test  = X[test_idx, :]
y_train_bubble = y_bubble[train_idx]
y_test_bubble  = y_bubble[test_idx]
y_train_dew    = y_dew[train_idx]
y_test_dew     = y_dew[test_idx]

println("Training: $(size(X_train, 1)) rows")
println("Testing:  $(size(X_test, 1)) rows")

# Custom operators
square(x::T) where {T} = x*x
cube(x::T) where {T} = x*x*x
inv_(x::T) where {T} = one(T)/x
invsquare(x::T) where {T} = one(T)/(x*x)
safesqrt(x::T) where {T} = x >= zero(T) ? sqrt(x) : T(NaN)
safelog(x::T) where {T} = x > zero(T) ? log(x) : T(NaN)

# Shared options
options = Options(;
    binary_operators = (+, -, *, /),
    unary_operators  = (square, cube, inv_, invsquare, safesqrt, safelog, abs),
    populations      = 40,
    maxsize          = 30,
    parsimony        = 0.001f0,
)

# Run SR for P_bubble
println("\n══ Fitting P_bubble ══")
hall_of_fame_bubble = equation_search(
    X_train', y_train_bubble;   # SymbolicRegression expects (n_features, n_samples)
    options   = options,
    niterations = 100,
    variable_names = features,
    parallelism = :multithreading,
)

println("\nBest equations for P_bubble:")
println(hall_of_fame_bubble)

# Run SR for P_dew
println("\n══ Fitting P_dew ══")
hall_of_fame_dew = equation_search(
    X_train', y_train_dew;
    options   = options,
    niterations = 100,
    variable_names = features,
    parallelism = :multithreading,
)

println("\nBest equations for P_dew:")
println(hall_of_fame_dew)

serialize("outputs/hall_of_fame_bubble.jls", hall_of_fame_bubble)
serialize("outputs/hall_of_fame_dew.jls", hall_of_fame_dew)