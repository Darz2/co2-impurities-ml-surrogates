# export JULIA_NUM_THREADS=2
# Usage:  julia Mixture_source1.jl [source_id]
# Default source_id = 1

SOURCE_ID = isempty(ARGS) ? 1 : parse(Int, ARGS[1])

println("CPU threads available = ", Sys.CPU_THREADS)
println("Julia threads         = ", Threads.nthreads())
println("source_id             = ", SOURCE_ID)

using CSV, DataFrames, Random, Statistics, Plots, Serialization, SymbolicRegression, JSON3
using Base.Threads
cd(@__DIR__)
outdir = "outputs/source$(SOURCE_ID)"
mkpath(outdir)
include("sr_utils.jl")

# Load Critical properties of pure fluids from PC-SAFT + cDFT
crit = JSON3.read(read("../Pure_fluid/component_Tc.json", String))
Tc_map = Dict(string(k) => Float64(v["Tc"]) for (k, v) in crit if string(k) != "carbon_dioxide")
Pc_map = Dict(string(k) => Float64(v["Pc"]) for (k, v) in crit if string(k) != "carbon_dioxide") # pc in bar

# CO2 critical properties (used as base-fluid Tr/Pr features)
Tc_CO2 = Float64(crit["carbon_dioxide"]["Tc"])   # [K]
Pc_CO2 = Float64(crit["carbon_dioxide"]["Pc"])   # [bar]

# Load data and filter by source_id
df_all = CSV.read("1691761_CombinedDataset_A3.csv", DataFrame; normalizenames=true)
df     = filter(row -> row.source_id == SOURCE_ID, df_all)
println("Rows for source_id = $(SOURCE_ID): ", nrow(df))

# ===========================================
# CO2 baseline (WSD) -- eps_base -- Target
# ===========================================
r1                  = (df[!, :rhoL_carbon_dioxide] .- df[!, :rhoV_carbon_dioxide]) ./
                        (df[!, :rhoL0_carbon_dioxide] .- df[!, :rhoV0_carbon_dioxide])
df[!, :gamma_base]  = df[!, :gamma0_carbon_dioxide] .* r1 .^ 2
df[!, :gamma_cDFT]  = df[!, :gamma_wsd] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]
df[!, :eps_base]    = (df[!, :gamma_cDFT] .- df[!, :gamma_base]) ./ df[!, :gamma_base]

# ===========================================
# Read the impurities in the dataframe
# ===========================================
x_imp_cols_all = [c for c in names(df) if startswith(c, "x_") && c != "x_carbon_dioxide"]
imp_names_all  = replace.(x_imp_cols_all, "x_" => "")

# Keep only impurities that are actually present (non-zero x) in this source_id
imp_names  = filter(imp -> any(df[!, Symbol("x_" * imp)] .> 0), imp_names_all)
x_imp_cols = ["x_" * imp for imp in imp_names]
y_imp_cols = ["y_" * imp for imp in imp_names]
z_imp_cols = ["z_" * imp for imp in imp_names]

println("Impurity components in source $(SOURCE_ID) (", length(imp_names), "): ", join(imp_names, ", "))

missing_Tc = [n for n in imp_names if !haskey(Tc_map, n)]
missing_Pc = [n for n in imp_names if !haskey(Pc_map, n)]
isempty(missing_Tc) || error("Missing Tc for: $(join(missing_Tc, ", "))")
isempty(missing_Pc) || error("Missing Pc for: $(join(missing_Pc, ", "))")

# ============================================================
# Feature engineering
# CO2 base-fluid Tr / Pr features (mirrors parity_plot_full.jl)
# ============================================================

df[!, :Tr_CO2]         = df[!, :T] ./ Tc_CO2
df[!, :Pr_CO2]         = df[!, :P] ./ Pc_CO2
df[!, :Pr_over_Tr_CO2] = df[!, :Pr_CO2] ./ df[!, :Tr_CO2]
df[!, :x_imp_sum]      = sum(df[!, Symbol("x_" * imp)] for imp in imp_names_all)
df[!, :y_imp_sum]      = sum(df[!, Symbol("y_" * imp)] for imp in imp_names_all)
df[!, :phi]            = (df[!, :rho_l] .- df[!, :rho_v]) ./ (df[!, :rho_l] .+ df[!, :rho_v])
df[!, :phi2]            = df[!, :phi].^2
df[!, :phi3]            = df[!, :phi].^3
df[!, :phi4]            = df[!, :phi].^4

df[!, :x_phi0] = df[!, :x_imp_sum]
df[!, :x_phi1] = df[!, :x_imp_sum] .* df[!, :phi]
df[!, :x_phi2] = df[!, :x_imp_sum] .* df[!, :phi2]
df[!, :x_phi3] = df[!, :x_imp_sum] .* df[!, :phi3]
df[!, :x_phi4] = df[!, :x_imp_sum] .* df[!, :phi4]

# ============================================================
# Feature engineering
# Per-impurity reduced-pressure and reduced-temperature features
# ============================================================

# Pressure-scaled features
x_pr_features = Symbol[]
y_pr_features = Symbol[]

# Reduced-temperature features
x_tr_features = Symbol[]
y_tr_features = Symbol[]

# Distance-above-critical features: (T/Tc - 1)
x_trm1_features = Symbol[]
y_trm1_features = Symbol[]

for imp in imp_names
    xcol = Symbol("x_" * imp)
    ycol = Symbol("y_" * imp)

    Tr_i   = df[!, :T] ./ Tc_map[imp]
    Pr_i   = df[!, :P] ./ Pc_map[imp]
    Trm1_i = Tr_i .- 1.0

    # Names
    fxPr   = Symbol("x_" * imp * "_Pr")
    fyPr   = Symbol("y_" * imp * "_Pr")

    fxTr   = Symbol("x_" * imp * "_Tr")
    fyTr   = Symbol("y_" * imp * "_Tr")

    fxTrm1 = Symbol("x_" * imp * "_Trm1")
    fyTrm1 = Symbol("y_" * imp * "_Trm1")

    # Build columns
    df[!, fxPr]   = df[!, xcol] .* Pr_i
    df[!, fyPr]   = df[!, ycol] .* Pr_i

    df[!, fxTr]   = df[!, xcol] .* Tr_i
    df[!, fyTr]   = df[!, ycol] .* Tr_i

    df[!, fxTrm1] = df[!, xcol] .* Trm1_i
    df[!, fyTrm1] = df[!, ycol] .* Trm1_i

    # Store feature names separately
    push!(x_pr_features, fxPr)
    push!(y_pr_features, fyPr)

    push!(x_tr_features, fxTr)
    push!(y_tr_features, fyTr)

    push!(x_trm1_features, fxTrm1)
    push!(y_trm1_features, fyTrm1)
end

# ================================
# Target and feature selection
# ================================

target      = :eps_base
# features    = unique(vcat([:x_imp_sum, :phi, :phi2, :phi3, :phi4]))
features = [:x_phi0, :x_phi1, :x_phi2, :x_phi3, :x_phi4, :x_imp_sum, :phi, :phi2, :phi3, :phi4]

println("\nTarget:   ", target)
println("Features: ", features)

X = Matrix{Float64}(df[:, features])
y = Vector{Float64}(df[:, target])

# ============================================================
# Train / validation / test split  (70 / 15 / 15)
# ============================================================

n   = size(X, 1)
rng = MersenneTwister(5002034)
idx = randperm(rng, n)

n_train = round(Int, 0.70 * n)
n_val   = round(Int, 0.15 * n)

train_idx = idx[1:n_train]
val_idx   = idx[n_train+1:n_train+n_val]
test_idx  = idx[n_train+n_val+1:end]

X_train, X_val, X_test = X[train_idx,:], X[val_idx,:], X[test_idx,:]
y_train, y_val, y_test = y[train_idx],   y[val_idx],   y[test_idx]

println("Train / Val / Test = $(size(X_train,1)) / $(size(X_val,1)) / $(size(X_test,1))")


# ============================================================
# Symbolic regression
# ============================================================

square(x) = x^2

# options = Options(
#     binary_operators = [+, -, *, /, safepow],
#     unary_operators  = [log, tanh, safesqrt, safe_erfc, one_plus, one_plus_square, square, cube],
#     complexity_of_operators = Dict(tanh => 5, safesqrt =>2, safe_erfc => 2, one_plus => 1, one_plus_square => 2, square => 2, cube => 3),
#     maxsize = 30,
#     populations = max(Threads.nthreads(), 36),
#     population_size = 160,
#     batching = true,
#     batch_size = 256,
#     parsimony = 0.001f0,
# )

options = Options(
    binary_operators        = [+, *],
    unary_operators         = [square],
    populations             = 200,
    maxsize                 = 20,
    parsimony               = 0.01f0,
    complexity_of_constants = 2,
)

println("\n══ Fitting eps_base  (source_id = $(SOURCE_ID)) ══")

hof = equation_search(
    X_train', y_train;
    options        = options,
    niterations    = 500,
    variable_names = string.(features),
    parallelism    = :multithreading,
)

serialize("$outdir/hall_of_fame_eps_base.jls", hof)

# ============================================================
# Select best equation from Pareto frontier (lowest val RMSE)
# ============================================================

dominating = calculate_pareto_frontier(hof)
best_idx, _ = select_best_equation(dominating, X_val, y_val, options)
best_member  = dominating[best_idx]
eq_str       = string_tree(best_member.tree, options)

println("\nSelected equation [$best_idx]: ", eq_str)

# ============================================================
# Evaluate on all splits
# ============================================================

yhat_train, _ = eval_tree_array(best_member.tree, X_train', options)
yhat_val,   _ = eval_tree_array(best_member.tree, X_val',   options)
yhat_test,  _ = eval_tree_array(best_member.tree, X_test',  options)

print_metrics("Train", y_train, yhat_train)
print_metrics("Val",   y_val,   yhat_val)
print_metrics("Test",  y_test,  yhat_test)

# ============================================================
# Parity plot
# ============================================================

parity_plot(
    y_train, yhat_train,
    y_val,   yhat_val,
    y_test,  yhat_test;
    title    = "eps_base  |  source_id=$(SOURCE_ID)",
    xlabel   = "Actual ε_base",
    ylabel   = "Predicted ε_base",
    eq_idx   = best_idx,
    savepath = "$outdir/parity_eps_base.png",
)

# ============================================================
# Save Pareto equations to CSV
# ============================================================

eq_rows = []
for (i, member) in enumerate(dominating)
    eq_s  = string_tree(member.tree, options)
    comp  = compute_complexity(member, options)

    yhat_tr, ok_tr = eval_tree_array(member.tree, X_train', options)
    yhat_vl, ok_vl = eval_tree_array(member.tree, X_val',   options)
    yhat_te, ok_te = eval_tree_array(member.tree, X_test',  options)

    push!(eq_rows, (
        complexity  = comp,
        score       = member.score,
        loss        = member.loss,
        selected    = (i == best_idx),
        equation    = eq_s,
        rmse_train  = ok_tr ? rmse(y_train, yhat_tr) : NaN,
        rmse_val    = ok_vl ? rmse(y_val,   yhat_vl) : NaN,
        rmse_test   = ok_te ? rmse(y_test,  yhat_te) : NaN,
        r2_train    = ok_tr ? r2_score(y_train, yhat_tr) : NaN,
        r2_val      = ok_vl ? r2_score(y_val,   yhat_vl) : NaN,
        r2_test     = ok_te ? r2_score(y_test,  yhat_te) : NaN,
    ))
end

eq_df = DataFrame(eq_rows)
CSV.write("$outdir/pareto_equations.csv", eq_df)
println("Saved Pareto equations → $outdir/pareto_equations.csv")