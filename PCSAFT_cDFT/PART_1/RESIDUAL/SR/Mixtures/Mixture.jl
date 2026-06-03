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
using Parquet2
using JSON3

cd(@__DIR__)
const OUTDIR = "SR_MIXTURES_OUTPUTS"
mkpath(OUTDIR)
include("sr_utils.jl")

mae(y, yhat) = mean(abs.(y .- yhat))

# safe_div appears in trained equations but is not handled by sr_utils.string_to_math
function expand_safe_div(eq_str::AbstractString)
    s = String(eq_str)
    for _ in 1:200
        result = extract_function_args(s, "safe_div", true)
        result === nothing && break
        arg1, arg2, start, end_pos = result
        arg1 = strip_outer_parens(arg1)
        arg2 = strip_outer_parens(arg2)
        s = s[1:start-1] * "(($arg1) / ($arg2))" * s[end_pos+1:end]
    end
    return s
end

# ============================================================
# Data loading and feature engineering
# ============================================================
df                  = CSV.read("../../CombinedDatasetSEC_A4.csv", DataFrame; normalizenames=true)
df[!, :gamma_base]  = df[!, :gamma_wsd_UC]   # uncorrected WSD model
df[!, :gamma_cDFT]  = df[!, :gamma_wsd_UC] .+ df[!, :gamma_cDFT_minus_wsd_uncorrected]
# ============================================================
# Symbolic Regression targets
# ============================================================
# Target : Relative residual vs baseline
df[!, :eps_base] = (df[!, :gamma_cDFT] .- df[!, :gamma_base]) ./ df[!, :gamma_base]

# ============================================================
# Dimensionless candidate features 
# ============================================================

# Dynamic impurity lists — all components except CO2
x_imp_cols = [c for c in names(df) if startswith(c, "x_") && c != "x_carbon_dioxide"]
y_imp_cols = [c for c in names(df) if startswith(c, "y_") && c != "y_carbon_dioxide"]
z_imp_cols = [c for c in names(df) if startswith(c, "z_") && c != "z_carbon_dioxide"]

println("\nImpurity components (", length(x_imp_cols), "): ",
        join(replace.(x_imp_cols, "x_" => ""), ", "))

# Total impurity sums (liquid, vapor, feed)
df[!, :x_total_impurity] = reduce(.+, [df[!, c] for c in x_imp_cols])
df[!, :y_total_impurity] = reduce(.+, [df[!, c] for c in y_imp_cols])
df[!, :z_total_impurity] = reduce(.+, [df[!, c] for c in z_imp_cols])

# Total impurity phase-partitioning difference
df[!, :delta_total_impurity] = df[!, :x_total_impurity] .- df[!, :y_total_impurity]

# Impurity partition coefficient
eps = 1e-12
df[!, :logK_total_impurity] = log.((df[!, :y_total_impurity] .+ eps) ./ (df[!, :x_total_impurity] .+ eps))

# ============================================================
# Physics-informed features
# ============================================================
# State variables reduced by pure-CO2 critical / saturation references.
# Surface-tension residuals scale strongly with proximity to the critical
# point and with the density gap (Macleod–Sugden / Parachor variable).

df[!, :Tr]               = df[!, :T] ./ df[!, :Tc_carbon_dioxide]                       # reduced T (CO2 reference)
df[!, :oneMinusTr]       = 1.0 .- df[!, :Tr]                                            # (1 − Tr): Guggenheim scaling base
df[!, :Pr]               = df[!, :P] ./ df[!, :Pc_carbon_dioxide]                       # reduced P (CO2 reference)
df[!, :P_over_Psat0]     = df[!, :P] ./ df[!, :Psat0_carbon_dioxide]                    # proximity to pure-CO2 saturation
df[!, :delta_rho]        = df[!, :rho_l] .- df[!, :rho_v]                               # phase density gap (Parachor variable)
df[!, :delta_rho_norm]   = (df[!, :rho_l] .- df[!, :rho_v]) ./
                          ((df[!, :rhoL0_carbon_dioxide] .- df[!, :rhoV0_carbon_dioxide]) .+ eps)  # vs pure CO2
df[!, :gamma_base_ratio] = df[!, :gamma_wsd_UC] ./ (df[!, :gamma0_carbon_dioxide] .+ eps)          # mixture WSD / pure-CO2 γ

# ============================================================
# Target and features for Symbolic Regression
# ============================================================

target = :eps_base

# Compact first-pass feature set + physics-informed additions
features = [
    # Composition aggregates
    :z_total_impurity,
    :x_total_impurity,
    :y_total_impurity,
    :delta_total_impurity,
    :logK_total_impurity,
    # State (reduced by pure-CO2 references)
    :Tr,
    :oneMinusTr,
    :Pr,
    :P_over_Psat0,
    # Density gap (Parachor)
    :delta_rho,
    :delta_rho_norm,
    # Baseline-relative
    :gamma_base_ratio,
]

println("\nTarget used: ", target)
println("Features used: ", features)

n_before = nrow(df)
dropmissing!(df, vcat(features, [target]))
all_cols = vcat(features, [target])
filter!(row -> all(isfinite(row[c]) for c in all_cols), df)
n_after = nrow(df)
if n_before > n_after
    println("Dropped $(n_before - n_after) rows with missing/Inf/NaN values ($(n_after) remaining)")
end

X          = Matrix{Float64}(df[:, features])
y          = Vector{Float64}(df[:, target])
gamma_base = Vector{Float64}(df[:, :gamma_base])

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

gamma_base_train = gamma_base[train_idx]
gamma_base_val   = gamma_base[val_idx]
gamma_base_test  = gamma_base[test_idx]

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
    binary_operators        = (+, -, *, safepow, safe_div),
    unary_operators         = (abs, safe_sqrt, safe_log),
    populations             = 120,
    maxsize                 = 24,
    parsimony               = 0.003f0,
    complexity_of_operators = [
        safe_div  => 2,
        safepow   => 3,
        abs       => 2,
        safe_sqrt => 2,
        safe_log  => 2,
    ],
    complexity_of_constants = 2,
    batching                = true,
    batch_size              = 1000,
)

# ============================================================
# Train symbolic regression
# ============================================================

println("\n══ Fitting residual correction eps_base ══")

hall_of_fame_eps_base = equation_search(
    X_train', y_train;
    options        = options,
    niterations    = 200,
    variable_names = string.(features),
    parallelism    = :multithreading,
)

serialize(joinpath(OUTDIR, "hall_of_fame_eps_base.jls"), hall_of_fame_eps_base)

println("\nSaved hall of fame to:")
println(joinpath(OUTDIR, "hall_of_fame_eps_base.jls"))

println("\nHall of fame summary:")
println(hall_of_fame_eps_base)

# ============================================================
# Choose best equation using validation RMSE
# ============================================================

dominating_eps_base = calculate_pareto_frontier(hall_of_fame_eps_base)
best_idx_auto, best_rmse_val_auto = select_best_equation(dominating_eps_base, X_val, y_val, options)

# Select the equation at a target complexity (robust to frontier reordering).
# We choose the complexity-12 rational form — the "knee" of the Pareto
# frontier: ln(Δρ*) / [√Pr + (γ_r − Δρ*)]. Set TARGET_COMPLEXITY = nothing to
# fall back to the lowest-validation-RMSE auto-pick.
TARGET_COMPLEXITY = 12
target_idx = TARGET_COMPLEXITY === nothing ? nothing :
    findfirst(m -> compute_complexity(m, options) == TARGET_COMPLEXITY, dominating_eps_base)
best_idx = target_idx === nothing ? best_idx_auto : target_idx
println("\nUsing best_idx = $best_idx  (auto-pick = $best_idx_auto, val RMSE = $best_rmse_val_auto, target complexity = $TARGET_COMPLEXITY)")

best_eq_eps_base = dominating_eps_base[best_idx]
best_eq_str = string_tree(best_eq_eps_base.tree, options)

println("\nChosen equation index = ", best_idx)
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

mae_train = mae(y_train, yhat_train)
mae_val   = mae(y_val,   yhat_val)
mae_test  = mae(y_test,  yhat_test)

# Reconstructed gamma_cDFT predictions (mN/m) — comparable to GPR/SVGP metrics
gamma_cDFT_actual_train = gamma_base_train .* (1.0 .+ y_train)
gamma_cDFT_actual_val   = gamma_base_val   .* (1.0 .+ y_val)
gamma_cDFT_actual_test  = gamma_base_test  .* (1.0 .+ y_test)

gamma_cDFT_pred_train = gamma_base_train .* (1.0 .+ yhat_train)
gamma_cDFT_pred_val   = gamma_base_val   .* (1.0 .+ yhat_val)
gamma_cDFT_pred_test  = gamma_base_test  .* (1.0 .+ yhat_test)

gamma_rmse_train = rmse(gamma_cDFT_actual_train, gamma_cDFT_pred_train)
gamma_rmse_val   = rmse(gamma_cDFT_actual_val,   gamma_cDFT_pred_val)
gamma_rmse_test  = rmse(gamma_cDFT_actual_test,  gamma_cDFT_pred_test)

gamma_r2_train = r2_score(gamma_cDFT_actual_train, gamma_cDFT_pred_train)
gamma_r2_val   = r2_score(gamma_cDFT_actual_val,   gamma_cDFT_pred_val)
gamma_r2_test  = r2_score(gamma_cDFT_actual_test,  gamma_cDFT_pred_test)

gamma_mae_train = mae(gamma_cDFT_actual_train, gamma_cDFT_pred_train)
gamma_mae_val   = mae(gamma_cDFT_actual_val,   gamma_cDFT_pred_val)
gamma_mae_test  = mae(gamma_cDFT_actual_test,  gamma_cDFT_pred_test)

# ============================================================
# Save predictions (incl. reconstructed gamma_cDFT for cross-model overlay)
# ============================================================

CSV.write(joinpath(OUTDIR, "SR_train_predictions_eps_base.csv"),
    DataFrame(
        Actual            = y_train,
        Predicted         = yhat_train,
        gamma_base        = gamma_base_train,
        gamma_cDFT_actual = gamma_cDFT_actual_train,
        gamma_cDFT_pred   = gamma_cDFT_pred_train,
    ))
CSV.write(joinpath(OUTDIR, "SR_validation_predictions_eps_base.csv"),
    DataFrame(
        Actual            = y_val,
        Predicted         = yhat_val,
        gamma_base        = gamma_base_val,
        gamma_cDFT_actual = gamma_cDFT_actual_val,
        gamma_cDFT_pred   = gamma_cDFT_pred_val,
    ))
CSV.write(joinpath(OUTDIR, "SR_test_predictions_eps_base.csv"),
    DataFrame(
        Actual            = y_test,
        Predicted         = yhat_test,
        gamma_base        = gamma_base_test,
        gamma_cDFT_actual = gamma_cDFT_actual_test,
        gamma_cDFT_pred   = gamma_cDFT_pred_test,
    ))

println("\nSaved prediction tables in $OUTDIR")

# ============================================================
# Save metrics JSON (schema compatible with GPR/SVGP loaders)
# ============================================================

best_complexity = compute_complexity(best_eq_eps_base, options)
best_loss       = best_eq_eps_base.loss

eq_raw   = best_eq_str
eq_math  = string_to_math(expand_safe_div(eq_raw))
eq_latex = math_to_latex(eq_math)

metrics = Dict(
    "model"                   => "SR_eps_base",
    "target"                  => String(target),
    "features"                => String.(features),
    "seed"                    => 5002034,
    "n_train"                 => length(y_train),
    "n_val"                   => length(y_val),
    "n_test"                  => length(y_test),
    # eps_base (dimensionless) metrics
    "train_rmse"              => rmse_train,
    "val_rmse"                => rmse_val,
    "test_rmse"               => rmse_test,
    "train_r2"                => r2_train,
    "val_r2"                  => r2_val,
    "test_r2"                 => r2_test,
    "train_mae"               => mae_train,
    "val_mae"                 => mae_val,
    "test_mae"                => mae_test,
    # Reconstructed gamma_cDFT (mN/m) metrics — comparable to GPR/SVGP
    "gamma_cDFT_rmse_train"   => gamma_rmse_train,
    "gamma_cDFT_rmse_val"     => gamma_rmse_val,
    "gamma_cDFT_rmse_test"    => gamma_rmse_test,
    "gamma_cDFT_r2_train"     => gamma_r2_train,
    "gamma_cDFT_r2_val"       => gamma_r2_val,
    "gamma_cDFT_r2_test"      => gamma_r2_test,
    "gamma_cDFT_mae_train_mNm"=> gamma_mae_train,
    "gamma_cDFT_mae_val_mNm"  => gamma_mae_val,
    "gamma_cDFT_mae_test_mNm" => gamma_mae_test,
    # Equation
    "equation_index"          => best_idx,
    "equation_raw"            => eq_raw,
    "equation_math"           => eq_math,
    "equation_latex"          => eq_latex,
    "complexity"              => best_complexity,
    "sr_loss"                 => best_loss,
)

open(joinpath(OUTDIR, "SR_residual_metrics.json"), "w") do io
    JSON3.pretty(io, metrics)
end

println("\nSaved metrics JSON:")
println(joinpath(OUTDIR, "SR_residual_metrics.json"))

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

savefig(p_eps_base, joinpath(OUTDIR, "SR_parity_plot_eps_base.png"))
savefig(p_eps_base, joinpath(OUTDIR, "SR_parity_plot_eps_base.pdf"))

println("\nSaved parity plot:")
println(joinpath(OUTDIR, "SR_parity_plot_eps_base.png"))
println(joinpath(OUTDIR, "SR_parity_plot_eps_base.pdf"))