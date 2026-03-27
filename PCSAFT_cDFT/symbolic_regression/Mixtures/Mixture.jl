# export JULIA_NUM_THREADS=2

println("CPU threads available = ", Sys.CPU_THREADS)
println("Julia threads         = ", Threads.nthreads())

using CSV
using DataFrames
using Random
using Statistics
using Plots
using Serialization
using SymbolicRegression

cd(@__DIR__)
mkpath("outputs")

include("sr_utils.jl")

# ============================================================
# Data loading and feature engineering
# ============================================================

df      = CSV.read("CSV/interfacial_results/PT_wsd.csv", DataFrame; normalizenames=true)
df_cDFT = CSV.read("CSV/interfacial_results/feed_1_interfacial_results.csv", DataFrame; normalizenames=true)
# df = first(df, 5)

# Mixture CO2 density gap
df[!, :NUM1]        = df[!, :rhoL1_carbon_dioxide] .- df[!, :rhoV1_carbon_dioxide]

# Pure CO2 density gap
df[!, :DEN1]        = df[!, :rhoL0_carbon_dioxide] .- df[!, :rhoV0_carbon_dioxide]
df[!, :C]           = 1 .- (df[!, :x_hydrogen] .+ df[!, :x_argon])                        # Correction

# Reference density-gap ratio
df[!, :r1]          = df[!, :NUM1] ./ df[!, :DEN1]
df[!, :r1sq]        = df[!, :r1] .^ 2

# WSD baseline if only CO2 contributes
# gamma_base = gamma0_CO2 * r1^2
df[!, :gamma_base] = df[!, :gamma0_carbon_dioxide] .* df[!, :r1sq]

# Optional supercritical correction used in your previous workflow
df[!, :C] = 1 .- (df[!, :x_hydrogen] .+ df[!, :x_argon])
df[!, :gamma_base_corr] = df[!, :gamma_base] .* df[!, :C]

println(all(isapprox.(df.gamma_wsd, df.gamma_base_corr; atol=1e-12, rtol=1e-10)) ?
    "gamma_wsd matches gamma_base_corr within the chosen tolerances." :
    "gamma_wsd does not match gamma_base_corr within the chosen tolerances.")

# cDFT Gamma values
# gamma_cDFT_minus_wsd_uncorrected = gamma_cDFT - gamma_wsd/gamma_base_corr
df[!, :gamma_cDFT] = df[!, :gamma_wsd] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]


# ============================================================
# Symbolic Regression targets
# ============================================================

# Target : Relative residual vs baseline
df[!, :eps_base] = (df[!, :gamma_cDFT] .- df[!, :gamma_base]) ./ df[!, :gamma_base]

# Optional target: absolute residual vs baseline
df[!, :delta_gamma_base] = df[!, :gamma_cDFT] .- df[!, :gamma_base]


# ============================================================
# Dimensionless candidate features 
# ============================================================

# Phase-partitioning differences
df[!, :delta_z_hydrogen] = df[!, :x_hydrogen] .- df[!, :y_hydrogen]
df[!, :delta_z_argon]    = df[!, :x_argon]    .- df[!, :y_argon]

# Component density gaps
df[!, :delta_rho_hydrogen] = df[!, :rhoL2_hydrogen] .- df[!, :rhoV2_hydrogen]
df[!, :delta_rho_argon]    = df[!, :rhoL3_argon]    .- df[!, :rhoV3_argon]

# Ratios relative to CO2 mixture density gap
df[!, :q_hydrogen] = df[!, :delta_rho_hydrogen] ./ df[!, :NUM1]
df[!, :q_argon]    = df[!, :delta_rho_argon]    ./ df[!, :NUM1]

# Ratios relative to pure CO2 reference density gap
df[!, :rtilde_hydrogen] = df[!, :delta_rho_hydrogen] ./ df[!, :DEN1]
df[!, :rtilde_argon]    = df[!, :delta_rho_argon]    ./ df[!, :DEN1]

# Bounded gap-share variables
df[!, :gap_total] = df[!, :NUM1] .+ df[!, :delta_rho_hydrogen] .+ df[!, :delta_rho_argon]
df[!, :s_hydrogen] = df[!, :delta_rho_hydrogen] ./ df[!, :gap_total]
df[!, :s_argon]    = df[!, :delta_rho_argon]    ./ df[!, :gap_total]

# Optional partition coefficients
eps = 1e-12
df[!, :logK_hydrogen] = log.((df[!, :y_hydrogen] .+ eps) ./ (df[!, :x_hydrogen] .+ eps))
df[!, :logK_argon]    = log.((df[!, :y_argon]    .+ eps) ./ (df[!, :x_argon]    .+ eps))

# Optional interaction terms
df[!, :qH_qAr] = df[!, :q_hydrogen] .* df[!, :q_argon]
df[!, :dzH_dzAr] = df[!, :delta_z_hydrogen] .* df[!, :delta_z_argon]

# Total impurity liquid fraction
df[!, :x_total_impurity] = df[!, :x_hydrogen] .+ df[!, :x_argon]

# Total impurity vapor fraction 
df[!, :y_total_impurity] = df[!, :y_hydrogen] .+ df[!, :y_argon]

# Total impurity phase-partitioning difference
df[!, :delta_z_total_impurity] = df[!, :x_total_impurity] .- df[!, :y_total_impurity]

# Impurity partition coefficient
df[!, :logK_total_impurity] = log.((df[!, :y_total_impurity] .+ eps) ./ (df[!, :x_total_impurity] .+ eps))

# Total impurity density gap
df[!, :q_total_impurity] =
    df[!, :delta_rho_total_impurity] ./
    (df[!, :rhoL1_carbon_dioxide] .- df[!, :rhoV1_carbon_dioxide])

# ============================================================
# Target and features for Symbolic Regression
# ============================================================

target = :eps_base

# Compact first-pass feature set
features = [
    :x_total_impurity,
    :y_total_impurity,
    :delta_z_total_impurity,
    :q_total_impurity,
]

println("\\nTarget used: ", target)
println("Features used: ", features)

X = Matrix{Float64}(df[:, features])
y = Vector{Float64}(df[:, target])


# ============================================================
# Train/validation/test split
# ============================================================

n = size(X, 1)
rng = MersenneTwister(5002034)
idx = randperm(rng, n)

n_train = round(Int, 0.70 * n)
n_val   = round(Int, 0.15 * n)

train_idx = idx[1:n_train]
val_idx   = idx[n_train+1:n_train+n_val]
test_idx  = idx[n_train+n_val+1:end]

X_train = X[train_idx, :]
X_val   = X[val_idx, :]
X_test  = X[test_idx, :]

y_train = y[train_idx]
y_val   = y[val_idx]
y_test  = y[test_idx]

println("Training rows   = ", size(X_train, 1))
println("Validation rows = ", size(X_val, 1))
println("Testing rows    = ", size(X_test, 1))

# ============================================================
# Symbolic regression options
# ============================================================

# options = Options(
#     binary_operators = (+, -, *, safepow),
#     populations      = 50,
#     maxsize          = 12,
#     parsimony        = 0.01f0,
# )

options = Options(
    binary_operators        = (+, -, *, safepow),
    unary_operators         = (abs,),
    populations             = 50,
    maxsize                 = 12,
    parsimony               = 0.01f0,
    complexity_of_operators = [safepow => 3, abs => 2],
    complexity_of_constants = 2
)

# ============================================================
# Train symbolic regression
# ============================================================

println("\n══ Fitting residual correction eps_base ══")

hall_of_fame_eps_base = equation_search(
    X_train', y_train;
    options        = options,
    niterations    = 500,
    variable_names = string.(features),
    parallelism    = :multithreading,
)

serialize("outputs/hall_of_fame_eps_base.jls", hall_of_fame_eps_base)

println("\nSaved hall of fame to:")
println("outputs/hall_of_fame_eps_base.jls")

println("\nHall of fame summary:")
println(hall_of_fame_eps_base)

# ============================================================
# Choose best equation using validation RMSE
# ============================================================

dominating_eps_base = calculate_pareto_frontier(hall_of_fame_eps_base)
# best_idx, best_rmse_val = select_best_equation(dominating_eps_base, X_val, y_val, options)

best_idx = 4
best_eq_eps_base = dominating_eps_base[best_idx]
best_eq_str = string_tree(best_eq_eps_base.tree, options)

println("\nChosen equation index (best validation RMSE) = ", best_idx)
# println("Best validation RMSE = ", best_rmse_val)
println("Chosen equation:")
println(best_eq_eps_base)

println("\nEquation string:")
println(best_eq_str)

# ============================================================
# Evaluate chosen equation
# ============================================================

yhat_train, ok_train = eval_tree_array(best_eq_eps_base.tree, X_train', options)
yhat_val,   ok_val   = eval_tree_array(best_eq_eps_base.tree, X_val',   options)
yhat_test,  ok_test  = eval_tree_array(best_eq_eps_base.tree, X_test',  options)

println("\nEvaluation successful (train): ", ok_train)
println("Evaluation successful (val):   ", ok_val)
println("Evaluation successful (test):  ", ok_test)

print_metrics("Train", y_train, yhat_train)
print_metrics("Validation", y_val, yhat_val)
print_metrics("Test", y_test, yhat_test)

rmse_train = rmse(y_train, yhat_train)
rmse_val   = rmse(y_val, yhat_val)
rmse_test  = rmse(y_test, yhat_test)

r2_train = r2_score(y_train, yhat_train)
r2_val   = r2_score(y_val, yhat_val)
r2_test  = r2_score(y_test, yhat_test)

# ============================================================
# Save predictions
# ============================================================

CSV.write("outputs/train_predictions_eps_base.csv", DataFrame(Actual=y_train, Predicted=yhat_train))
CSV.write("outputs/validation_predictions_eps_base.csv", DataFrame(Actual=y_val, Predicted=yhat_val))
CSV.write("outputs/test_predictions_eps_base.csv", DataFrame(Actual=y_test, Predicted=yhat_test))

println("\nSaved prediction tables:")
println("outputs/train_predictions_eps_base.csv")
println("outputs/validation_predictions_eps_base.csv")
println("outputs/test_predictions_eps_base.csv")

# ============================================================
# Parity plot
# ============================================================

xmin = min(
    minimum(y_train), minimum(yhat_train),
    minimum(y_val),   minimum(yhat_val),
    minimum(y_test),  minimum(yhat_test)
)

xmax = max(
    maximum(y_train), maximum(yhat_train),
    maximum(y_val),   maximum(yhat_val),
    maximum(y_test),  maximum(yhat_test)
)

p_eps_base = scatter(
    y_train, yhat_train;
    xlabel = "Actual ε / [mN/m]",
    ylabel = "Predicted ε / [mN/m]",
    label = "Train",
    markersize = 4,
    title = "Parity Plot for ε",
    legend = :topleft,
)

scatter!(p_eps_base, y_val,  yhat_val;  label = "Validation", markersize = 4)
scatter!(p_eps_base, y_test, yhat_test; label = "Test", markersize = 4)

plot!(
    p_eps_base,
    [xmin, xmax], [xmin, xmax];
    label = "y = x",
    lw = 2,
    color = :black,
    line = :dash,
)

annotate!(p_eps_base, xmin + 0.52*(xmax - xmin), xmax - 0.72*(xmax - xmin),
    text("Chosen Eq. index = $best_idx", 9, :left))
annotate!(p_eps_base, xmin + 0.52*(xmax - xmin), xmax - 0.78*(xmax - xmin),
    text("Train: RMSE = $(round(rmse_train, sigdigits=2)), R² = $(round(r2_train, sigdigits=2))", 9, :left))
annotate!(p_eps_base, xmin + 0.52*(xmax - xmin), xmax - 0.84*(xmax - xmin),
    text("Val: RMSE = $(round(rmse_val, sigdigits=2)), R² = $(round(r2_val, sigdigits=2))", 9, :left))
annotate!(p_eps_base, xmin + 0.52*(xmax - xmin), xmax - 0.90*(xmax - xmin),
    text("Test: RMSE = $(round(rmse_test, sigdigits=2)), R² = $(round(r2_test, sigdigits=2))", 9, :left))

savefig(p_eps_base, "outputs/parity_plot_eps_base.png")

println("\nSaved parity plot:")
println("outputs/parity_plot_eps_base.png")